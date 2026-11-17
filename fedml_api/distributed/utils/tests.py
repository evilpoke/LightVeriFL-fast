"""Tests of LightVeriAgg = LightSecAgg (aggregation) + LightVeriFL (verification, El Gamal commitment).

    python -m pytest -q fedml_api/distributed/utils/tests.py          # from the repository root
    python fedml_api/distributed/utils/tests.py                       # same

* crypto building blocks: EC helpers, El Gamal commitment (separate integer mask of the randomness),
  linearly homomorphic hash, (N, T+1) MDS codes over the curve and over Z_n
* LightSecAgg: Lagrange coefficients, LCC encoding/decoding (no int64 overflow), mask padding
* quantization, simulated dropouts, benchmarking helpers, the single-process message bus
* end-to-end runs of the protocol (single-process mode): honest server, verification dropouts,
  and malicious servers that forge the aggregate model or the reconstructed hash
"""
import argparse
import csv
import copy
import math
import os
import pickle
import random
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fastecdsa.curve import P256  # noqa: E402
from fastecdsa.point import Point  # noqa: E402

from fedml_api.distributed.lightveriagg.LightVeriAggAggregator import aggregate_mask_functioncall_reconstruction  # noqa: E402
from fedml_api.distributed.lightveriagg.LightVeriAggClientManager import LightVeriAggClientManager  # noqa: E402
from fedml_api.distributed.lightveriagg.LightVeriAggServerManager import LightVeriAggServerManager  # noqa: E402
from fedml_api.distributed.lightveriagg.message_define import MyMessage  # noqa: E402
from fedml_api.distributed.lightveriagg.mpc_function import compute_aggregate_encoded_mask, mask_encoding, \
    matmul_mod, model_masking, padded_dimension, random_field_array, verification_dropouts  # noqa: E402
from fedml_api.distributed.lightveriagg.single_process import FedML_lightveriagg_single_process, PHASES, \
    message_label  # noqa: E402
from fedml_api.distributed.lightveriagg.utils import finite_aggregate_to_averaged_tensor, finite_to_signed, \
    flatten_finite_model, gen_Lagrange_coeffs, gen_Lagrange_coeffs_lightsecagg, my_q, my_q_inv, \
    transform_tensor_to_finite  # noqa: E402
from fedml_api.distributed.utils.EC import IDENTITY, LightVeriFL_dec_EC, LightVeriFL_dec_scalar, \
    LightVeriFL_enc_EC, LightVeriFL_enc_scalar, PI_addEC, add_field_elements, aggregate_el_gamal_commitments, \
    compute_elgamal_commitment_to_hash_randomness, curve_g, curve_n, ec_add, ec_mul, elgamal_commitments_equal, \
    generate_point_EC, get_hash_bases, hash_to_point, hash_vector, is_identity, random_scalar, \
    veri_eval_points  # noqa: E402
from fedml_api.distributed.utils.bench import Bench, fmt_bits, measure, num_bits, timer  # noqa: E402
from fedml_core.distributed.communication.local import LocalBus  # noqa: E402
from fedml_core.distributed.communication.message import Message  # noqa: E402
from fedml_api.model.linear.lr import LogisticRegression  # noqa: E402
from fedml_api.model.cv.cifar10_models import CIFAR10_MODELS  # noqa: E402

p_model = 2 ** 31 - 1      # field of the (quantized) models, default of main_lightveriagg.py
q_bits = 10                # precision_parameter, default of main_lightveriagg.py


# =====================================================================================================
# the original tests (completed)
# =====================================================================================================
def test1():
    # a local LightSecAgg mask z_i: d uniformly random elements of F_p
    p = 2147483647
    d = int(600372.0)
    local_mask = random_field_array(p, (d, 1))
    assert local_mask.shape == (d, 1) and local_mask.dtype == np.int64
    assert local_mask.min() >= 0 and local_mask.max() < p
    assert len(np.unique(local_mask[:1000])) > 990          # not constant / not seeded to the same values
    assert not np.array_equal(local_mask, random_field_array(p, (d, 1)))


def different_generators():
    from fedml_api.distributed.utils.EC import different_generators as _different_generators
    return _different_generators()


def test_different_generators():
    # "nothing up my sleeve": G and H = hash_to_point(...) are distinct points on P-256 and H is
    # reproducible by everybody, but nobody knows log_G(H) (with H = 6G, H = 3 * (2G) the commitment
    # could be opened by everybody).
    g, h = different_generators()
    assert P256.is_point_on_curve((g.x, g.y)) and P256.is_point_on_curve((h.x, h.y))
    assert g != h and h == hash_to_point(b"LightVeriAgg/ElGamal-commitment/H")
    assert all(h != k * g for k in range(1, 100))


def test2():
    # el gamal thingy: every client masks its hash with an EC point and the randomness of its El Gamal
    # commitment with a separate integer in Z_n. The demasked aggregates open the aggregate commitment.
    total_dimension = 44
    bases = get_hash_bases(total_dimension)

    gradients, hashes, randomnesses, commitments = [], [], [], []
    masked_hashes, masked_randomnesses, zi_for_hashes, zi_for_randomnesses = [], [], [], []

    for i in range(10):  # 10 virtual clients
        weights = {"something": np.random.uniform(-2, 2, size=(total_dimension, 1))}
        gradient = flatten_finite_model(transform_tensor_to_finite(weights, p_model, q_bits), p_model)
        hash_ = hash_vector(gradient, bases)
        randomness = random_scalar()

        zi_for_hash = generate_point_EC()                        # group element mask of the hash
        zi_for_randomness = random_scalar()                      # field element mask of the randomness
        masked_hash = ec_add(hash_, zi_for_hash)
        masked_randomness = add_field_elements([randomness, zi_for_randomness], curve_n)

        gradients.append(gradient)
        hashes.append(hash_)
        randomnesses.append(randomness)
        commitments.append(compute_elgamal_commitment_to_hash_randomness(hash_, randomness))
        masked_hashes.append(masked_hash)
        masked_randomnesses.append(masked_randomness)
        zi_for_hashes.append(zi_for_hash)
        zi_for_randomnesses.append(zi_for_randomness)

    # aggregate the masks, the masked hashes and the masked randomness
    aggregated_mask_for_hashes = PI_addEC(zi_for_hashes)
    aggregated_mask_for_randomness = add_field_elements(zi_for_randomnesses, curve_n)
    aggregated_masked_hashes = PI_addEC(masked_hashes)
    aggregated_masked_randomness = add_field_elements(masked_randomnesses, curve_n)

    # demask
    demasked_aggregated_randomness = (aggregated_masked_randomness - aggregated_mask_for_randomness) % curve_n
    demasked_aggregated_hashes = ec_add(aggregated_masked_hashes, -aggregated_mask_for_hashes)

    assert demasked_aggregated_hashes == PI_addEC(hashes)
    assert demasked_aggregated_randomness == add_field_elements(randomnesses, curve_n)
    # the hash of the aggregated gradient is the aggregate of the hashes
    assert hash_vector(sum(gradients), bases) == PI_addEC(hashes)
    # the aggregate commitment opens to the demasked aggregates
    assert elgamal_commitments_equal(
        aggregate_el_gamal_commitments(commitments),
        compute_elgamal_commitment_to_hash_randomness(demasked_aggregated_hashes, demasked_aggregated_randomness))


# =====================================================================================================
# elliptic curve helpers
# =====================================================================================================
def test_points_survive_pickling():
    # every MPI message is pickled; fastecdsa compares/adds points with `curve is curve`
    P = generate_point_EC()
    Q = pickle.loads(pickle.dumps(P))
    assert Q.curve is P256 and Q == P
    assert Q + P == ec_mul(2, P)
    assert is_identity(pickle.loads(pickle.dumps(IDENTITY)))


def test_identity_and_scalar_edge_cases():
    P = generate_point_EC()
    assert is_identity(ec_mul(0, P)) and is_identity(ec_mul(5, IDENTITY)) and is_identity(ec_mul(curve_n, P))
    assert is_identity(PI_addEC([])) and PI_addEC([IDENTITY, P, IDENTITY]) == P
    assert is_identity(PI_addEC([P, -P]))
    assert ec_mul(-3, P) == -ec_mul(3, P) and ec_mul(np.int64(7), P) == 7 * P
    assert ec_mul(curve_n + 2, P) == ec_mul(2, P)
    assert ec_add(IDENTITY, P) == P == ec_add(P, IDENTITY)


def test_random_scalars_and_points_are_not_seeded():
    # main_lightveriagg.py seeds random/np.random/torch with 0 in every process: crypto randomness must not depend on it
    random.seed(0)
    np.random.seed(0)
    a = (random_scalar(), generate_point_EC())
    random.seed(0)
    np.random.seed(0)
    b = (random_scalar(), generate_point_EC())
    assert a[0] != b[0] and a[1] != b[1]
    assert 0 < a[0] < curve_n


# =====================================================================================================
# El Gamal commitment
# =====================================================================================================
def test_elgamal_commitment_is_homomorphic():
    hs = [generate_point_EC() for _ in range(5)]
    rs = [random_scalar() for _ in range(5)]
    cs = [compute_elgamal_commitment_to_hash_randomness(h, r) for h, r in zip(hs, rs)]
    assert elgamal_commitments_equal(aggregate_el_gamal_commitments(cs),
                                     compute_elgamal_commitment_to_hash_randomness(PI_addEC(hs), sum(rs) % curve_n))


def test_elgamal_commitment_is_binding_and_randomized():
    h, h2 = generate_point_EC(), generate_point_EC()
    r = random_scalar()
    c = compute_elgamal_commitment_to_hash_randomness(h, r)
    assert c[0] == ec_mul(r, curve_g)                          # c1 = rG determines r -> binding for h
    assert not elgamal_commitments_equal(c, compute_elgamal_commitment_to_hash_randomness(h2, r))
    assert not elgamal_commitments_equal(c, compute_elgamal_commitment_to_hash_randomness(h, (r + 1) % curve_n))
    # without H, c2 - r G' would reveal h: make sure c2 is not h + r * (small multiple of G)
    assert all(ec_add(c[1], -ec_mul(k, c[0])) != h for k in range(1, 20))


def test_elgamal_commitment_of_empty_aggregate():
    # D = [N] \ U_1 is usually empty: the empty aggregates (identity, 0) must verify
    empty = aggregate_el_gamal_commitments([])
    assert elgamal_commitments_equal(empty, compute_elgamal_commitment_to_hash_randomness(IDENTITY, 0))
    assert is_identity(empty[0]) and is_identity(empty[1])


# =====================================================================================================
# linearly homomorphic hash
# =====================================================================================================
def test_hash_is_linear_for_signed_vectors():
    d = 30
    bases = get_hash_bases(d)
    xs = [np.random.randint(-5000, 5000, size=d) for _ in range(4)]
    assert hash_vector(sum(xs), bases) == PI_addEC([hash_vector(x, bases) for x in xs])
    assert hash_vector(3 * xs[0], bases) == ec_mul(3, hash_vector(xs[0], bases))
    assert is_identity(hash_vector(np.zeros(d, dtype=np.int64), bases))
    big = [int(v) * (2 ** 70) for v in xs[0]]                                # big (batch check) coefficients
    assert hash_vector(big, bases) == ec_mul(2 ** 70, hash_vector(xs[0], bases))


def test_hash_bases_are_deterministic_distinct_points():
    bases = get_hash_bases(25)
    assert get_hash_bases(25) is bases                                      # cached per process
    assert len({(g.x, g.y) for g in bases}) == 25
    assert all(P256.is_point_on_curve((g.x, g.y)) for g in bases)
    assert bases[:10] == get_hash_bases(10)                                 # independent of d


def test_hash_of_field_aggregate_equals_sum_of_hashes():
    # The models are aggregated in F_p (p = 2^31-1) but the hash is linear modulo the group order n.
    # Negative weights are encoded as p - |x| and sums wrap around mod p; reading the field elements as
    # signed integers (centered lift) keeps h(y) = sum_i h(x_i).
    d, N = 40, 6
    bases = get_hash_bases(d)
    models = [{"w": np.random.uniform(-1, 1, size=(d,))} for _ in range(N)]
    finite = [transform_tensor_to_finite(copy.deepcopy(m), p_model, q_bits) for m in models]
    y = {"w": np.mod(sum(f["w"] for f in finite), p_model)}
    assert np.any(sum(f["w"] for f in finite) >= p_model)                   # the plain sum did wrap around
    hashes = [hash_vector(flatten_finite_model(f, p_model), bases) for f in finite]
    assert hash_vector(flatten_finite_model(y, p_model), bases) == PI_addEC(hashes)
    # ... whereas hashing the representatives in [0, p) does not work
    naive = [hash_vector(f["w"], bases) for f in finite]
    assert hash_vector(y["w"], bases) != PI_addEC(naive)


# =====================================================================================================
# MDS codes of the verification masks (EC points and integers)
# =====================================================================================================
def test_gen_lagrange_coeffs_with_256_bit_prime():
    # used to fail with "OverflowError: Python int too large to convert to C long"
    alpha_s, beta_s = veri_eval_points(6, 3)
    W = gen_Lagrange_coeffs(alpha_s, beta_s, curve_n)
    coeffs = [random.randrange(curve_n) for _ in range(len(alpha_s))]
    f = lambda x: sum(c * pow(x, k, curve_n) for k, c in enumerate(coeffs)) % curve_n
    for i, b in enumerate(beta_s):
        assert sum(int(W[i][j]) * f(a) for j, a in enumerate(alpha_s)) % curve_n == f(b)


@pytest.mark.parametrize("N,T", [(3, 1), (5, 2), (8, 4)])
def test_verification_masks_one_shot_decoding(N, T):
    evalpoints_in, evalpoints_out = veri_eval_points(N, T)
    zh = [generate_point_EC() for _ in range(N)]
    zr = [random_scalar() for _ in range(N)]
    enc_h = [LightVeriFL_enc_EC(z, [generate_point_EC() for _ in range(T)], evalpoints_in, evalpoints_out, P256)
             for z in zh]
    enc_r = [LightVeriFL_enc_scalar(z, [random_scalar() for _ in range(T)], evalpoints_in, evalpoints_out)
             for z in zr]
    U = sorted(random.sample(range(N), k=max(T + 1, N - 1)))          # aggregate over a subset of users
    for responders in (list(range(T + 1)), sorted(random.sample(range(N), T + 1))):
        agg_h = [PI_addEC([enc_h[j][i] for j in U]) for i in responders]
        agg_r = [sum(enc_r[j][i] for j in U) % curve_n for i in responders]
        points = [evalpoints_out[i] for i in responders]
        assert LightVeriFL_dec_EC(agg_h, evalpoints_in, points, P256) == PI_addEC([zh[j] for j in U])
        assert LightVeriFL_dec_scalar(agg_r, evalpoints_in, points) == sum(zr[j] for j in U) % curve_n


def test_verification_mask_encoding_needs_T_plus_1_points():
    evalpoints_in, evalpoints_out = veri_eval_points(5, 2)
    with pytest.raises(AssertionError):
        LightVeriFL_enc_EC(generate_point_EC(), [generate_point_EC()], evalpoints_in, evalpoints_out, P256)
    with pytest.raises(AssertionError):
        LightVeriFL_dec_scalar([1, 2], evalpoints_in, evalpoints_out[:3])


def _solve_mod(A, b, q):
    """solve A x = b over Z_q (A square, invertible) by Gauss-Jordan elimination"""
    n = len(A)
    M = [[int(v) % q for v in row] + [int(bi) % q] for row, bi in zip(A, b)]
    for col in range(n):
        piv = next(r for r in range(col, n) if M[r][col])
        M[col], M[piv] = M[piv], M[col]
        inv = pow(M[col][col], -1, q)
        M[col] = [v * inv % q for v in M[col]]
        for r in range(n):
            if r != col and M[r][col]:
                f = M[r][col]
                M[r] = [(a - f * c) % q for a, c in zip(M[r], M[col])]
    return [M[r][n] for r in range(n)]


@pytest.mark.parametrize("T", [1, 2, 3])
def test_T_shares_of_a_verification_mask_reveal_nothing(T):
    # T-privacy of the (N, T+1) code: the shares of any T colluding users can be explained by ANY other
    # mask (with suitable blindings), i.e. they carry no information about the integer mask zr_i
    N = T + 3
    evalpoints_in, evalpoints_out = veri_eval_points(N, T)
    secret, blindings = random_scalar(), [random_scalar() for _ in range(T)]
    shares = LightVeriFL_enc_scalar(secret, blindings, evalpoints_in, evalpoints_out)
    colluders = sorted(random.sample(range(N), T))
    W = gen_Lagrange_coeffs(evalpoints_in, evalpoints_out, curve_n)
    other_secret = random_scalar()
    rhs = [(shares[i] - int(W[i][0]) * other_secret) % curve_n for i in colluders]
    other_blindings = _solve_mod([[W[i][j + 1] for j in range(T)] for i in colluders], rhs, curve_n)
    other_shares = LightVeriFL_enc_scalar(other_secret, other_blindings, evalpoints_in, evalpoints_out)
    assert [other_shares[i] for i in colluders] == [shares[i] for i in colluders]
    assert other_shares != shares                       # ... while T+1 shares determine the mask


# =====================================================================================================
# LightSecAgg
# =====================================================================================================
def test_matmul_mod_does_not_overflow():
    rng = np.random.default_rng(0)
    A = rng.integers(0, p_model, size=(7, 12))
    B = rng.integers(0, p_model, size=(12, 50))
    exact = np.mod(A.astype(object).dot(B.astype(object)), p_model).astype(np.int64)
    assert np.array_equal(matmul_mod(A, B, p_model), exact)
    assert not np.array_equal(np.mod(A.dot(B), p_model), exact)          # what np.dot did before
    big_p = 2 ** 61 - 1
    A2, B2 = A % 1000, B % 1000
    assert np.array_equal(matmul_mod(A2, B2, big_p), A2.dot(B2) % big_p)


def test_gen_lagrange_coeffs_lightsecagg_interpolates():
    N, U = 7, 5
    alpha_s = np.array(range(U)) + (N + 1)
    beta_s = np.array(range(N)) + 1
    W = gen_Lagrange_coeffs_lightsecagg(beta_s, alpha_s, p_model)
    coeffs = [random.randrange(p_model) for _ in range(U)]
    f = lambda x: sum(c * pow(int(x), k, p_model) for k, c in enumerate(coeffs)) % p_model
    for i, b in enumerate(beta_s):
        assert sum(int(W[i][j]) * f(a) for j, a in enumerate(alpha_s)) % p_model == f(b)


@pytest.mark.parametrize("N,U,T,d", [(5, 5, 2, 30730), (10, 8, 3, 1001), (20, 12, 5, 777), (4, 3, 1, 10)])
def test_lightsecagg_mask_reconstruction(N, U, T, d):
    # each user encodes its mask, the users in U_1 aggregate the shares, the server decodes sum z_i from
    # the aggregated shares of any U users -- also for d not divisible by U-T and U large (int64 overflow)
    d_pad = padded_dimension(d, U, T)
    assert d_pad % (U - T) == 0 and d <= d_pad < d + (U - T)
    masks = [random_field_array(p_model, (d_pad, 1)) for _ in range(N)]
    encoded = [mask_encoding(d_pad, N, U, T, p_model, z) for z in masks]
    U1 = sorted(random.sample(range(N), k=max(U, N - 1)))
    received = {i: {j: encoded[j][i] for j in range(N)} for i in range(N)}   # what user i holds
    aggregated = {i: compute_aggregate_encoded_mask(received[i], p_model, U1) for i in range(N)}
    responders = sorted(random.sample(range(N), k=U))
    decoded = aggregate_mask_functioncall_reconstruction(d, N, U, T, p_model, responders, aggregated)
    expected = np.mod(sum(masks[j] for j in U1), p_model)[:d]
    assert np.array_equal(decoded, expected)


def test_model_masking_and_aggregate_model_recovery():
    N, U, T = 4, 4, 1
    models = [{"a": np.random.uniform(-1, 1, size=(3, 4)), "b": np.random.uniform(-1, 1, size=(5,))}
              for _ in range(N)]
    finite = [transform_tensor_to_finite(copy.deepcopy(m), p_model, q_bits) for m in models]
    dims = [12, 5]
    d = sum(dims)
    d_pad = padded_dimension(d, U, T)
    masks = [random_field_array(p_model, (d_pad, 1)) for _ in range(N)]
    masked = [model_masking(copy.deepcopy(f), dims, z, p_model) for f, z in zip(finite, masks)]
    assert not np.array_equal(masked[0]["a"], finite[0]["a"])
    agg_mask = np.mod(sum(masks), p_model)[:d]
    y = {k: np.mod(sum(m[k] for m in masked) - agg_mask[o:o + n].reshape(masked[0][k].shape), p_model)
         for k, o, n in (("a", 0, 12), ("b", 12, 5))}
    avg = finite_aggregate_to_averaged_tensor(y, p_model, q_bits, N)
    for k in ("a", "b"):
        assert np.allclose(avg[k].numpy(), np.mean([m[k] for m in models], axis=0), atol=2 ** -q_bits)


# =====================================================================================================
# quantization
# =====================================================================================================
def test_quantization_roundtrip_and_centered_lift():
    x = np.array([-3.25, -1e-3, 0.0, 0.5, 7.125])
    xq = my_q(x, q_bits, p_model)
    assert xq.min() >= 0 and xq.max() < p_model
    assert np.allclose(my_q_inv(xq, q_bits, p_model), x, atol=2 ** -q_bits)
    assert np.array_equal(finite_to_signed(xq, p_model), np.round(x * 2 ** q_bits).astype(np.int64))


def test_quantization_of_float32_torch_weights():
    # torch weights are float32; in float32 p = 2^31-1 is not representable and negative weights used to be
    # quantized to ~2^31 (out of the field, "+1" after the centered lift): the models lost their negative weights
    w = torch.tensor([-0.0316, -1e-3, 0.0, 0.5, -3.25, 1000.0, -1000.0], dtype=torch.float32)
    finite = transform_tensor_to_finite({"w": w.clone()}, p_model, q_bits)["w"]
    assert finite.min() >= 0 and finite.max() < p_model
    assert np.allclose(my_q_inv(finite, q_bits, p_model), w.numpy(), atol=2 ** -q_bits)
    assert np.array_equal(finite_to_signed(finite, p_model), np.round(w.numpy().astype(np.float64) * 2 ** q_bits))


# =====================================================================================================
# simulated verification dropouts
# =====================================================================================================
def test_verification_dropouts():
    U1 = list(range(10))
    assert verification_dropouts(3, U1, 0.0, 4) == []
    drop = verification_dropouts(3, U1, 0.3, 4, seed=7)
    assert len(drop) == 3 and set(drop) <= set(U1)
    assert drop == verification_dropouts(3, U1, 0.3, 4, seed=7)                 # server and clients agree
    assert len(verification_dropouts(3, U1, 0.9, 4)) == 10 - 5                    # T+1 = 5 must survive
    assert any(verification_dropouts(r, U1, 0.3, 4) != drop for r in range(10))  # changes over the rounds


# =====================================================================================================
# benchmarking + single-process bus
# =====================================================================================================
def test_num_bits():
    P = generate_point_EC()
    assert num_bits(P) == 512 and num_bits(IDENTITY) == 8
    assert num_bits(curve_n - 1) == 256 and num_bits(255) == 8 and num_bits(-1) == 8 and num_bits(256) == 16
    assert num_bits(np.array([0, p_model - 1], dtype=np.int64)) == 64              # field elements: 32 bit
    assert num_bits(np.zeros(10, dtype=np.float32)) == 320 and num_bits(torch.zeros(4)) == 128
    assert num_bits([P, (P, 7)]) == 512 + 512 + 8 and num_bits({"k": np.zeros(2, np.int8)}) == 8 + 16
    with pytest.raises(TypeError):
        num_bits(object())


def test_measure_and_timer_record_into_the_active_bench():
    assert measure(np.zeros(4, dtype=np.float32)) == 128                          # no active bench
    bench = Bench()
    with bench:
        bench.set_round(1)
        measure(np.zeros(4, dtype=np.float32), "upload", src=1, dst="server")
        measure(np.zeros(8, dtype=np.float32), "upload", src=2, dst="server", rnd=2)
        with timer("client: work", who=1):
            pass
        with timer("server: work", rnd=5):
            pass
    measure(np.zeros(4), "ignored")
    assert [r["round"] for r in bench.comm] == [1, 2] and [r["round"] for r in bench.time] == [1, 5]
    (row,) = bench.comm_summary()
    assert row["bits"] == 384 and row["direction"] == "client->server"
    text = bench.report()
    assert "COMMUNICATION" in text and "COMPUTATION" in text and "ESTIMATED ROUND LATENCY" in text
    assert fmt_bits(8 * 1500) == "1.50 kB"


def test_local_bus_is_fifo_and_copies_messages():
    seen = []

    class Recorder:
        def receive_message(self, msg_type, msg):
            seen.append((msg_type, msg.get("payload")))

    sizes = []
    bus = LocalBus(on_send=lambda params, s, r: sizes.append((s, r)))
    for rank in (0, 1):
        bus.endpoint(rank).add_observer(Recorder())
    payload = {"x": [1, 2]}
    for t in (1, 2, 3):
        m = Message(t, 0, 1)
        m.add_params("payload", payload)
        bus.endpoints[0].send_message(m)
    payload["x"].append(99)                     # the sender changes its object after sending
    bus.run()
    assert [t for t, _ in seen] == [1, 2, 3] and seen[0][1] == {"x": [1, 2]}
    assert sizes == [(0, 1)] * 3 and bus.delivered == 3


# =====================================================================================================
# end-to-end: the protocol of message_define.py in single-process mode
# =====================================================================================================
def make_args(tmp_path, N=4, T=1, rounds=2, L=2, alpha_bits=64, dropout=0.0, **kw):
    args = argparse.Namespace(
        model="lr", dataset="synthetic", data_dir="", partition_method="homo", partition_alpha=0.5,
        client_num_in_total=N, client_num_per_round=N, batch_size=16, client_optimizer="sgd", backend="SP",
        lr=0.1, wd=0.0, epochs=1, comm_round=rounds, is_mobile=0, frequency_of_the_test=1, ci=0,
        targeted_number_active_clients=N, privacy_guarantee=T, prime_number=p_model, precision_parameter=q_bits,
        veri_batch_size=L, veri_alpha_bits=alpha_bits, veri_dropout_rate=dropout, veri_dropout_seed=0,
        single_process=1, results_dir=str(tmp_path), bench_name="test", uplink_mbps=10.0, downlink_mbps=50.0)
    for k, v in kw.items():
        setattr(args, k, v)
    return args


def run_protocol(args, server_class=LightVeriAggServerManager, client_class=LightVeriAggClientManager,
                 write_results=False, d_in=6, classes=3, samples=32, model=None, x_shape=None):
    """synthetic data (inputs of shape x_shape, default (d_in,)) and LogisticRegression(d_in, classes) or model"""
    torch.manual_seed(0)
    N = args.client_num_per_round
    x_shape = tuple(x_shape or (d_in,))
    X = torch.randn(N * samples, *x_shape)
    y = (X.flatten(1) @ torch.randn(int(np.prod(x_shape)), classes)).argmax(1)
    train_local = {i: DataLoader(TensorDataset(X[i * samples:(i + 1) * samples], y[i * samples:(i + 1) * samples]),
                                 batch_size=args.batch_size, shuffle=True) for i in range(N)}
    test_global = DataLoader(TensorDataset(X, y), batch_size=64)
    model = LogisticRegression(d_in, classes) if model is None else model
    return FedML_lightveriagg_single_process(
        torch.device("cpu"), model, N * samples, test_global, test_global, {i: samples for i in range(N)},
        train_local, {i: test_global for i in range(N)}, args,
        server_manager_class=server_class, client_manager_class=client_class, write_results=write_results)


class RecordingClient(LightVeriAggClientManager):
    """remembers the quantized local models x_i that it masked and uploaded"""
    uploaded = dict()

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        RecordingClient.uploaded[(self.round_idx, self.client_idx)] = copy.deepcopy(self.stored_model)
        super().send_model_to_server(receive_id, weights, local_sample_num)


class RecordingServer(LightVeriAggServerManager):
    """remembers the aggregate y = sum x_i that it broadcasts"""
    broadcast = dict()

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index):
        RecordingServer.broadcast[self.round_idx] = copy.deepcopy(global_model_params)
        super().send_message_sync_model_to_client(receive_id, global_model_params, client_index)


def test_protocol_end_to_end(tmp_path):
    RecordingClient.uploaded.clear()
    RecordingServer.broadcast.clear()
    args = make_args(tmp_path, N=4, T=1, rounds=3, L=2)
    result = run_protocol(args, server_class=RecordingServer, client_class=RecordingClient, write_results=True)
    summary = result["summary"]
    assert summary["all_rounds_verified"] and summary["rounds_completed"] == 3
    log = [rec for c in result["clients"] for rec in c.verification_log]
    assert len(log) == 4 * 3 and all(rec["commitments_ok"] for rec in log)
    # batch checks after round 1 (rounds 0, 1) and in the last round (round 2)
    assert sorted(tuple(rec["batch_checked_rounds"]) for rec in log if "batch_ok" in rec) == [(0, 1)] * 4 + [(2,)] * 4
    # LightSecAgg: the server recovered exactly y = sum_i x_i mod p
    for rnd in range(3):
        y = RecordingServer.broadcast[rnd]
        for k in y:
            expected = np.mod(sum(np.asarray(RecordingClient.uploaded[(rnd, c)][k], dtype=np.int64)
                                  for c in range(4)), p_model)
            assert np.array_equal(y[k], expected)
    # every message type of message_define.py was used, in rounds 0..2
    labels = {r["label"] for r in result["bench"].comm}
    types = [v for k, v in vars(MyMessage).items() if k.startswith("MSG_TYPE_")]
    assert labels == {message_label(t) for t in types}
    assert {r["round"] for r in result["bench"].comm} == {0, 1, 2}
    # benchmark output
    out = result["output_dir"]
    for name in ("report.txt", "metrics.json", "bench.json", "communication.png", "computation_per_step.png",
                 "computation_per_round.png", "convergence.png"):
        assert (out / name).exists()
    assert "all verified: True" in (out / "report.txt").read_text()
    check_csv_logs(out, result["bench"], args)


def check_csv_logs(out, bench, args):
    """the CSV logs are complete and consistent with the benchmark records"""
    read = lambda name: list(csv.DictReader(open(out / name, newline="")))  # noqa: E731
    run = read("run.csv")
    assert len(run) == 1 and int(run[0]["N"]) == args.client_num_per_round and run[0]["all_verified"] == "1"
    computation, communication = read("computation.csv"), read("communication.csv")
    assert len(computation) == len(bench.time) and len(communication) == len(bench.comm)
    # every timed step and every message type belongs to a phase of the protocol
    assert {r["phase"] for r in computation + communication} <= set(PHASES) - {"other"}
    rounds = read("rounds.csv")
    assert [int(r["round"]) for r in rounds] == list(range(args.comm_round))
    assert sum(int(r["comm_%s_bits" % ph]) for r in rounds for ph in PHASES) == sum(r["bits"] for r in bench.comm)
    per_round = bench.per_round_compute()
    for r in rounds:
        assert abs(float(r["time_iteration_s"]) - sum(per_round[int(r["round"])].values())) < 1e-9
        assert abs(float(r["time_protocol_s"]) + float(r["time_training_s"]) + float(r["time_evaluation_s"])
                   - float(r["time_iteration_s"])) < 1e-9
        assert r["test_acc"] != ""
    assert sum(int(r["batch_check"]) for r in rounds) == len({int(r["round"]) for r in computation
                                                               if r["phase"] == "batch_check"})
    assert len(read("verification.csv")) == args.client_num_per_round * args.comm_round


@pytest.mark.parametrize("name", list(CIFAR10_MODELS))
def test_cifar10_models_run_the_protocol(tmp_path, name):
    """the models of --model (cnn6k, cnn15k, lr, lenet5, mlp99k): 5k - 100k parameters, only parameters
    in the state_dict (no batch-norm buffers), two verified rounds on CIFAR-10-shaped inputs"""
    model = CIFAR10_MODELS[name](10)
    n_params = sum(p.numel() for p in model.parameters())
    assert 5000 <= n_params <= 100000
    assert sum(v.numel() for v in model.state_dict().values()) == n_params
    result = run_protocol(make_args(tmp_path, N=3, T=1, rounds=2, L=2, alpha_bits=0), model=model,
                          x_shape=(3, 32, 32), classes=10, samples=8)
    assert result["summary"]["all_rounds_verified"]
    assert result["server"].aggregator.total_dimension == n_params


def test_message_sizes(tmp_path):
    result = run_protocol(make_args(tmp_path, N=3, T=1, rounds=1, L=1))
    per_msg = {r["label"]: r["bits_per_message"] for r in result["bench"].comm_summary()}
    point, scalar = 512, 256

    def bits(t):
        return per_msg[message_label(t)]

    assert bits(MyMessage.MSG_TYPE_C2S_VERI_COMMITMENT_TO_SERVER) == 2 * point
    assert point + 200 <= bits(MyMessage.MSG_TYPE_C2S_VERI_MASKED_HASH_R) <= point + scalar   # h~ and r~
    assert bits(MyMessage.MSG_TYPE_C2S_VERI_SEND_MASK_TO_SERVER) <= 2 * (point + scalar)
    d = 6 * 3 + 3
    assert bits(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER) >= d * 32         # masked model: d field elements


@pytest.mark.parametrize("dropout", [0.4, 0.6])
def test_protocol_with_verification_dropouts(tmp_path, dropout):
    # N = 5, T = 2: up to 2 users may drop in every verification phase (T+1 = 3 shares are needed)
    args = make_args(tmp_path, N=5, T=2, rounds=3, L=2, dropout=dropout)
    result = run_protocol(args)
    assert result["summary"]["all_rounds_verified"]
    log = [rec for c in result["clients"] for rec in c.verification_log]
    assert sum(rec["dropped_in_verification"] for rec in log) == 2 * 3
    assert all(rec["commitments_ok"] for rec in log if not rec["dropped_in_verification"])


class ForgingServer(LightVeriAggServerManager):
    """malicious server: adds delta[round] to one coordinate of the aggregate model y"""
    delta = {}

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index):
        shift = ForgingServer.delta.get(self.round_idx, 0)
        if shift:
            global_model_params = copy.deepcopy(global_model_params)
            key = next(iter(global_model_params))
            flat = global_model_params[key].reshape(-1)
            flat[0] = (flat[0] + shift) % p_model
        super().send_message_sync_model_to_client(receive_id, global_model_params, client_index)


def test_forged_aggregate_model_is_detected(tmp_path):
    ForgingServer.delta = {1: 1000}
    result = run_protocol(make_args(tmp_path, N=4, T=1, rounds=2, L=2), server_class=ForgingServer)
    assert not result["summary"]["all_rounds_verified"]
    assert "Forged aggregation" in result["summary"]["verification_error"]


@pytest.mark.parametrize("alpha_bits,detected", [(0, False), (64, True)])
def test_cross_round_forgery_needs_random_batch_coefficients(tmp_path, alpha_bits, detected):
    # +delta in round 0 and -delta in round 1 cancel out in the plain sum of a batch (alpha_l = 1,
    # as in the LightVeriFL implementation); private random alpha_l (protocol of message_define.py) catch it
    ForgingServer.delta = {0: 1000, 1: -1000}
    result = run_protocol(make_args(tmp_path, N=4, T=1, rounds=2, L=2, alpha_bits=alpha_bits),
                          server_class=ForgingServer)
    assert result["summary"]["all_rounds_verified"] is (not detected)


class WrongHashServer(LightVeriAggServerManager):
    """malicious server: reconstructs the aggregate hash of the surviving users incorrectly"""

    def send_message_sync_veriagg_to_client(self, receive_id, aggregate_hash_U, aggregate_hash_D,
                                            aggregate_randomness_U, aggregate_randomness_D, client_index):
        super().send_message_sync_veriagg_to_client(receive_id, ec_add(aggregate_hash_U, curve_g), aggregate_hash_D,
                                                    aggregate_randomness_U, aggregate_randomness_D, client_index)


def test_forged_hash_reconstruction_is_detected(tmp_path):
    result = run_protocol(make_args(tmp_path, N=3, T=1, rounds=1, L=1), server_class=WrongHashServer)
    assert not result["summary"]["all_rounds_verified"]
    assert "commitment check failed" in result["summary"]["verification_error"]


# =====================================================================================================
# the baseline of the benchmark: LightVeriFL + SecAgg in one process
# =====================================================================================================
def test_lightverifl_secagg_baseline_simulation():
    from ServerVerification.LightVeriFL_SecAgg_single_process import LightVeriFLSecAggSimulation
    N, T, d = 5, 2, 20
    sim = LightVeriFLSecAggSimulation(N, T, d, L=2)
    bench = Bench().activate()
    try:
        for rnd in range(2):
            xs = [my_q(np.random.uniform(-1, 1, d), q_bits, p_model) for _ in range(N)]
            surviving = [i for i in range(N) if i not in verification_dropouts(rnd, range(N), 0.4, T)]
            assert len(surviving) == 3
            sim.run_round(rnd, xs, surviving)
    finally:
        bench.deactivate()
    log = sim.verification_log
    assert all(r["commitments_ok"] and r["aggregate_correct"] and r["hash_correct"] for r in log)
    assert log[0]["batch_ok"] is None and log[1]["batch_ok"] is True          # batch check after L = 2 rounds
    labels = {r["label"] for r in bench.time}
    assert "server: [A.3] SecAgg unmasking (SS decoding + PRG), recover y = sum x_i" in labels
    assert any("[V.1]" in label for label in labels)


def test_pedersen_variant_lets_a_server_forge_but_el_gamal_does_not(tmp_path):
    # LightVeriFL's commitment c = noise + g + h + l is not binding: a server that adds delta to the
    # aggregate model can shift the reconstructed hash by h(delta) and the noise by -h(delta) -- the
    # commitment check and the batch check of the users both pass.
    from ServerVerification.LightVeriFL_SecAgg_single_process import LightVeriFLSecAggSimulation
    N, T, d = 4, 1, 12
    sim = LightVeriFLSecAggSimulation(N, T, d, L=1)
    delta = np.zeros(d, dtype=np.int64)
    delta[3] = 777
    sim.forged_delta = {0: delta}
    xs = [my_q(np.random.uniform(-1, 1, d), q_bits, p_model) for _ in range(N)]
    sim.run_round(0, xs, surviving=list(range(N)))
    (rec,) = sim.verification_log
    assert rec["forged"] and not rec["aggregate_correct"]
    assert rec["commitments_ok"] and rec["batch_ok"]                     # forgery NOT detected

    # the same attack against LightVeriAgg: the El Gamal commitment (r G, h + r H) is binding, so no
    # randomness opens sum c_i to the shifted hash -> detected by the commitment check
    class AdaptiveForgingServer(LightVeriAggServerManager):
        delta = None

        def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index):
            global_model_params = copy.deepcopy(global_model_params)
            key = next(iter(global_model_params))
            global_model_params[key].reshape(-1)[0] = (global_model_params[key].reshape(-1)[0] + 777) % p_model
            AdaptiveForgingServer.delta = (key, 777)
            super().send_message_sync_model_to_client(receive_id, global_model_params, client_index)

        def send_message_sync_veriagg_to_client(self, receive_id, aggregate_hash_U, aggregate_hash_D,
                                                aggregate_randomness_U, aggregate_randomness_D, client_index):
            h_delta = ec_mul(777, get_hash_bases(1)[0])                    # h(delta), delta = 777 e_0
            super().send_message_sync_veriagg_to_client(receive_id, ec_add(aggregate_hash_U, h_delta), aggregate_hash_D,
                                                        aggregate_randomness_U, aggregate_randomness_D, client_index)

    result = run_protocol(make_args(tmp_path, N=4, T=1, rounds=1, L=1), server_class=AdaptiveForgingServer)
    assert not result["summary"]["all_rounds_verified"]
    assert "commitment check failed" in result["summary"]["verification_error"]


def test_shared_mask_of_the_baseline_leaks_individual_hashes():
    # In LightVeriFL(+SecAgg) the same EC point z_i masks the hash and the commitment noise:
    #   hz_i = h_i + z_i, noise_z_i = r_i + z_i (server)  and  c_i = r_i + g + h_i + l (every user).
    # The server and ONE colluding user get 2 h_i = (hz_i - noise_z_i) + (c_i - g - l): every
    # individual hash leaks (LightVeriFL's input privacy goal). LightVeriAgg masks h_i with an EC point
    # and the El Gamal randomness with an independent integer, and c_i = (r_i G, h_i + r_i H) is hiding.
    from ServerVerification.LightVeriFL_SecAgg_single_process import LightVeriFLSecAggSimulation
    N, T, d = 4, 1, 12
    sim = LightVeriFLSecAggSimulation(N, T, d, L=1)
    xs = [my_q(np.random.uniform(-1, 1, d), q_bits, p_model) for _ in range(N)]
    sim.run_round(0, xs, surviving=list(range(N)))
    tr = sim.last_transcript
    inv2 = pow(2, -1, curve_n)
    for i in range(N):
        two_h = ec_add(ec_add(tr["hz"][i], -tr["noise_z"][i]),
                       ec_add(tr["commitments"][i], -ec_add(tr["Pedersen_g"], tr["Pedersen_l"])))
        assert ec_mul(inv2, two_h) == tr["hashes"][i]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"] + sys.argv[1:]))
