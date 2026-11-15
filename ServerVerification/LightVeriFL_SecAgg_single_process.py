"""LightVeriFL + SecAgg (the baseline of the LightVeriFL paper) in ONE process, with benchmarking.

Single-process port of LightVeriFL_EC_fastecdsa_amortized.py (the MPI implementation of the paper):
the same phases, the same simplified SecAgg (PRG masks with seeds from my_pk_gen/my_key_agreement,
Shamir shares of b_u and s_sk), the same LightVeriFL verification with EC point masks z_i that mask
the hash AND the Pedersen-variant noise, the same Pedersen-variant commitment c = noise + g + h + l and
the same amortized batch check with alpha_l = 1 (plain sum over the L rounds).

Every party is simulated one after another; the steps are timed with ``timer`` and every message is
sized with ``measure`` (fedml_api/distributed/utils/bench.py), exactly as LightVeriAgg in its
single-process mode, so that the two protocols can be compared label by label.

Deviations from the MPI script (to compare *protocols*, not implementations):
* the hash uses the same implementation and bases as LightVeriAgg (hash_vector / get_hash_bases;
  the MPI script recomputes alpha_j * G inside every hash call) and the field elements are read as
  signed integers (centered lift; the script hashed the representatives in [0, p), which only works
  for its non-negative test vectors x_i = [rank] * d),
* messages that the users exchange directly in the MPI script (encoded masks, commitments) are relayed
  by the server (as in LightVeriAgg) and counted as upload + download,
* the local models x_i are given (e.g. the ones recorded from a LightVeriAgg run).
"""
import random

import numpy as np

from fastecdsa.curve import P256

from fedml_api.distributed.utils.EC import IDENTITY, LightVeriFL_dec_EC, LightVeriFL_enc_EC, PI_addEC, curve_g, \
    ec_add, ec_mul, generate_point_EC, get_hash_bases, hash_vector, veri_eval_points
from fedml_api.distributed.utils.bench import measure, timer
from fedml_api.distributed.lightveriagg.utils import finite_to_signed
from ServerVerification.utils.function import SS_decoding, SS_encoding, my_key_agreement, my_pk_gen

p_model = 2 ** 31 - 1


def generate_Pedersen_commitment(hash, Pedersen_g, Pedersen_l, noise):
    # c = noise + g + hash + l   (LightVeriFL's "variant of the Pedersen commitment")
    return ec_add(ec_add(noise, Pedersen_g), ec_add(hash, Pedersen_l))


class LightVeriFLSecAggSimulation(object):
    def __init__(self, N, T, d, L, p=p_model, seed=0):
        self.N, self.T, self.d, self.L, self.p = N, T, d, L, p
        self.rng = random.Random(seed)
        self.bases = get_hash_bases(d)
        self.alpha_s, self.beta_s = veri_eval_points(N, T)
        # setup (server): Pedersen coefficients (as in the MPI script: small random integers)
        self.Pedersen_coeff = np.array(self.rng.sample(range(1, 100), 2))
        self.server_agg_gradient_epochs = []
        self.agg_client_hashes_epochs = []
        self.verification_log = []
        # malicious server (for tests): {round: delta}. The server adds delta to the aggregate model and,
        # since c = noise + g + h + l is not binding, shifts the reconstructed hash by h(delta) and the
        # reconstructed noise by -h(delta): both checks of the users still pass.
        self.forged_delta = dict()

    def run_round(self, rnd, xs, surviving):
        """One iteration. xs[i]: local model of user i in F_p (1-D int64); surviving: users that stay
        online in the verification phase (the others drop after the aggregation phase)."""
        N, T, d, p = self.N, self.T, self.d, self.p
        surviving = sorted(surviving)
        dropped = [i for i in range(N) if i not in surviving]
        actual = sorted(self.rng.sample(surviving, T + 1))     # the T+1 users that answer in Round V.0
        Pedersen_g = ec_mul(int(self.Pedersen_coeff[0]), curve_g)
        Pedersen_l = ec_mul(int(self.Pedersen_coeff[1]), curve_g)

        # ---------------- Aggregation Round 0. AdvertiseKeys ----------------
        keys = []
        for u in range(N):
            with timer("client: [A.0] SecAgg: key generation", who=u + 1, rnd=rnd):
                my_sk = np.random.randint(0, p, size=(2)).astype('int64')
                my_pk = my_pk_gen(my_sk, p, 0)
            keys.append((my_pk, my_sk))
            measure(my_pk, "A.0 public keys (up)", src=u + 1, dst="server", rnd=rnd)
        public_key_list = np.stack([k[0] for k in keys], axis=1)            # (2, N)
        for u in range(N):
            measure(public_key_list, "A.0 public keys (down)", src="server", dst=u + 1, rnd=rnd)

        # ---------------- Aggregation Round 1. ShareMetadata ----------------
        z_tilde = []
        z_mask = []
        for u in range(N):
            with timer("client: [A.1] verification mask z_i: generate + encode (EC)", who=u + 1, rnd=rnd):
                z = generate_point_EC()
                n_array = [generate_point_EC() for _ in range(T)]
                z_tilde.append(LightVeriFL_enc_EC(z, n_array, self.alpha_s, self.beta_s, P256))
            z_mask.append(z)
            for v in range(N):
                if v != u:
                    measure(z_tilde[u][v], "A.1 encoded verification masks (up)", src=u + 1, dst="server", rnd=rnd)
                    measure(z_tilde[u][v], "A.1 encoded verification masks (down)", src="server", dst=v + 1, rnd=rnd)

        secagg_mask, ss_shares = [], []
        for u in range(N):
            rank = u + 1
            my_pk, my_sk = keys[u]
            with timer("client: [A.1] SecAgg: self + pairwise masks (PRG)", who=rank, rnd=rnd):
                s_pk_list = public_key_list[1, :]
                b_u = my_sk[0]
                s_uv = np.mod(s_pk_list * my_sk[1], p)
                mask = np.zeros(d, dtype='int64')
                for i in range(1, N + 1):
                    if rank == i:
                        np.random.seed(b_u)
                        temp = np.random.randint(0, p, size=d, dtype='int64')
                    elif rank > i:
                        np.random.seed(s_uv[i - 1])
                        temp = np.random.randint(0, p, size=d, dtype='int64')
                    else:
                        np.random.seed(s_uv[i - 1])
                        temp = -np.random.randint(0, p, size=d, dtype='int64')
                    mask = np.mod(mask + temp, p)
            with timer("client: [A.1] SecAgg: secret-share b_u and s_sk", who=rank, rnd=rnd):
                b_u_SS = SS_encoding(int(my_sk[0]), N, T, p)
                s_sk_SS = SS_encoding(int(my_sk[1]), N, T, p)
            secagg_mask.append(mask)
            ss_shares.append((np.array(b_u_SS, dtype='int64'), np.array(s_sk_SS, dtype='int64')))
            measure(list(ss_shares[u]), "A.1 secret shares (up)", src=rank, dst="server", rnd=rnd)
        for v in range(N):
            measure([np.array([ss_shares[u][0][v] for u in range(N)]), np.array([ss_shares[u][1][v] for u in range(N)])],
                    "A.1 secret shares (down)", src="server", dst=v + 1, rnd=rnd)

        hz, noise_z, commitments, hashes = [], [], [], []
        for u in range(N):
            with timer("client: [A.2] hash h_i = H(x_i)", who=u + 1, rnd=rnd):
                h = hash_vector(finite_to_signed(xs[u], p), self.bases)
            with timer("client: [A.2] Pedersen-variant commitment + mask h_i and noise", who=u + 1, rnd=rnd):
                Pedersen_noise = generate_point_EC()
                commitment = generate_Pedersen_commitment(h, Pedersen_g, Pedersen_l, Pedersen_noise)
                hz.append(ec_add(h, z_mask[u]))
                noise_z.append(ec_add(Pedersen_noise, z_mask[u]))
            hashes.append(h)
            commitments.append(commitment)
            measure(commitment, "A.2 commitment (up)", src=u + 1, dst="server", rnd=rnd)
            for v in range(N):
                if v != u:
                    measure(commitment, "A.2 commitment (down)", src="server", dst=v + 1, rnd=rnd)

        # ---------------- Aggregation Round 2. MaskedInputCollection ----------------
        y = []
        for u in range(N):
            with timer("client: [A.2] mask model x_i + SecAgg mask", who=u + 1, rnd=rnd):
                y.append(np.mod(np.asarray(xs[u], dtype='int64') + secagg_mask[u], p))
            measure([y[u], hz[u], noise_z[u]], "A.2 masked model + masked hash and noise (up)", src=u + 1,
                    dst="server", rnd=rnd)

        # ---------------- Aggregation Round 3. Unmasking ----------------
        drop_info = np.zeros((N,), dtype=int)       # no dropouts in the aggregation phase (as in the paper)
        SS_rx = np.empty((N, N), dtype='int64')
        for u in range(N):
            SS_info = np.array([ss_shares[i][0][u] if drop_info[i] == 0 else ss_shares[i][1][u] for i in range(N)])
            SS_rx[:, u] = SS_info
            measure(SS_info, "A.3 secret shares for unmasking (up)", src=u + 1, dst="server", rnd=rnd)
        with timer("server: [A.3] SecAgg unmasking (SS decoding + PRG), recover y = sum x_i", who="server", rnd=rnd):
            agg_grad = np.zeros(d, dtype='int64')
            for u in range(N):
                agg_grad = np.mod(agg_grad + y[u], p)
            for i in range(N):
                SS_input = np.reshape(SS_rx[i, 0:T + 1], (T + 1,)).tolist()
                b_u = SS_decoding(SS_input, range(T + 1), p)
                np.random.seed(b_u)
                temp = np.random.randint(0, p, size=d).astype('int64')
                agg_grad = np.mod(agg_grad - temp, p)
        delta = self.forged_delta.get(rnd)
        if delta is not None:
            agg_grad = np.mod(agg_grad + np.asarray(delta, dtype='int64'), p)
        for u in surviving:
            measure(agg_grad, "A.3 aggregate model (down)", src="server", dst=u + 1, rnd=rnd)

        # ---------------- Verification Round 0. AggregateDecommitting ----------------
        to_server_surviving, to_server_dropped = [], []
        for u in actual:
            with timer("client: [V.0] aggregate encoded verification masks", who=u + 1, rnd=rnd):
                z_tilde_mul_surviving = PI_addEC([z_tilde[i][u] for i in surviving])
                z_tilde_mul_dropped = PI_addEC([z_tilde[i][u] for i in dropped])
            to_server_surviving.append(z_tilde_mul_surviving)
            to_server_dropped.append(z_tilde_mul_dropped)
            measure([z_tilde_mul_surviving, z_tilde_mul_dropped], "V.0 aggregated encoded masks (up)", src=u + 1,
                    dst="server", rnd=rnd)
        with timer("server: [V.0] decode aggregate verification masks, unmask sum h, sum noise", who="server", rnd=rnd):
            hz_mul_surviving = PI_addEC([hz[x] for x in surviving])
            hz_mul_dropped = PI_addEC([hz[x] for x in dropped])
            noise_mul_surviving = PI_addEC([noise_z[x] for x in surviving])
            noise_mul_dropped = PI_addEC([noise_z[x] for x in dropped])
            points = [self.beta_s[i] for i in actual]
            dec_z = LightVeriFL_dec_EC(to_server_surviving, self.alpha_s, points, P256)
            res_surviving = ec_add(hz_mul_surviving, -dec_z)
            noise_surviving = ec_add(noise_mul_surviving, -dec_z)
            dec_z = LightVeriFL_dec_EC(to_server_dropped, self.alpha_s, points, P256)
            res_dropped = ec_add(hz_mul_dropped, -dec_z)
            noise_dropped = ec_add(noise_mul_dropped, -dec_z)
            res = ec_add(res_dropped, res_surviving)
            res_noise = ec_add(noise_dropped, noise_surviving)
        if delta is not None:
            h_delta = hash_vector(finite_to_signed(np.mod(delta, p), p), self.bases)
            res, res_noise = ec_add(res, h_delta), ec_add(res_noise, -h_delta)
        for u in surviving:
            measure([res, res_noise], "V.0 aggregate hash and noise (down)", src="server", dst=u + 1, rnd=rnd)

        ok_commit = True
        for u in surviving:
            with timer("client: [V.0] verify commitments", who=u + 1, rnd=rnd):
                commitments_agg = generate_Pedersen_commitment(res, Pedersen_g, Pedersen_l, res_noise)
                commitments_summed = ec_add(PI_addEC(commitments), ec_mul(-(N - 1), ec_add(Pedersen_g, Pedersen_l)))
                ok_commit &= commitments_agg == commitments_summed

        # ---------------- Verification Round 1. BatchChecking (every L rounds, alpha_l = 1) ----------------
        self.server_agg_gradient_epochs.append(agg_grad)
        self.agg_client_hashes_epochs.append(res)
        ok_batch = None
        if len(self.server_agg_gradient_epochs) == self.L:
            ok_batch = True
            for u in surviving:
                with timer("client: [V.1] batch check h(sum y(l)) == sum h^l", who=u + 1, rnd=rnd):
                    summed = sum(finite_to_signed(g, p).astype(object) for g in self.server_agg_gradient_epochs)
                    h_agg = hash_vector(summed, self.bases)
                    ok_batch &= h_agg == PI_addEC(self.agg_client_hashes_epochs)
            self.server_agg_gradient_epochs, self.agg_client_hashes_epochs = [], []

        # what the server (hz, noise_z, g, l) and one colluding user (all commitments) see -- for tests
        self.last_transcript = {"hz": hz, "noise_z": noise_z, "commitments": commitments, "Pedersen_g": Pedersen_g,
                                "Pedersen_l": Pedersen_l, "hashes": hashes}
        self.verification_log.append({"round": rnd, "commitments_ok": bool(ok_commit), "batch_ok": ok_batch,
                                      "aggregate_correct": bool(np.array_equal(
                                          agg_grad, np.mod(sum(np.asarray(x, dtype='int64') for x in xs), p))),
                                      "hash_correct": res == PI_addEC(hashes),
                                      "forged": delta is not None})
        return agg_grad
