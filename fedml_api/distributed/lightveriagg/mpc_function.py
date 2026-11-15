from fedml_api.distributed.lightveriagg.utils import gen_Lagrange_coeffs, gen_Lagrange_coeffs_lightsecagg
import numpy as np
import torch
import copy
import secrets

# Cryptographic masks must NOT come from the seeded global np.random state: main_lightveriagg.py seeds it
# with 0 in every process, which made all clients draw the very same masks.
_mask_rng = np.random.default_rng(secrets.randbits(128))


def random_field_array(p, size):
    """Uniformly random elements of F_p (int64) drawn from a cryptographically seeded generator."""
    return _mask_rng.integers(0, p, size=size, dtype=np.int64)


def matmul_mod(A, B, p):
    """(A @ B) mod p without int64 overflow (np.dot overflows as soon as two products ~p^2 are summed)."""
    A = np.asarray(A)
    B = np.asarray(B)
    if p >= 2 ** 31:  # products would not fit into int64: exact (slow) Python-int arithmetic
        return np.mod(A.astype(object).dot(B.astype(object)), p).astype(np.int64)
    A = np.mod(A.astype(np.int64), p)
    B = np.mod(B.astype(np.int64), p)
    out = np.zeros((A.shape[0], B.shape[1]), dtype=np.int64)
    for k in range(A.shape[1]):
        out = np.mod(out + np.outer(A[:, k], B[k, :]), p)  # every term < p^2 < 2^62
    return out


def padded_dimension(d, U, T):
    """Smallest multiple of U-T that is >= d (LightSecAgg splits the mask into U-T equal sub-masks)."""
    return int(np.ceil(float(d) / (U - T))) * (U - T)


def verification_dropouts(round_idx, active_clients, dropout_rate, T, seed=0):
    """Simulated dropouts of the verification phase (as in the LightVeriFL experiments).

    Every round, floor(dropout_rate * |U_1|) of the users in U_1 drop after having received the
    aggregate model: they do not upload their aggregated encoded verification masks (message 10)
    and do not verify. At least T+1 users must survive (the verification masks are decoded from
    T+1 shares). The choice is a deterministic function of (seed, round) so that the server knows
    how many uploads to wait for -- it plays the role of the server's time-out.
    """
    active = sorted(int(c) for c in active_clients)
    num = int(np.floor(float(dropout_rate) * len(active) + 1e-9))
    num = min(num, len(active) - (T + 1))
    if num <= 0:
        return []
    rng = np.random.RandomState((int(seed) * 1000003 + int(round_idx)) % (2 ** 32))
    return sorted(int(c) for c in rng.choice(active, size=num, replace=False))


def modular_inv(a, p):
    x, y, m = 1, 0, p
    while a > 1:
        q = a // m
        t = m

        m = np.mod(a, m)
        a = t
        t = y

        y, x = x - np.int64(q) * np.int64(y), t

        if x < 0:
            x = np.mod(x, p)
    return np.mod(x, p)


def divmod(_num, _den, _p):
    # compute num / den modulo prime p
    _num = np.mod(_num, _p)
    _den = np.mod(_den, _p)
    _inv = modular_inv(_den, _p)
    return np.mod(np.int64(_num) * np.int64(_inv), _p)


def PI_del(vals, p):  # upper-case PI -- product of inputs
    accum = np.int64(1)
    for v in vals:
        tmp = np.mod(v, p)
        accum = np.mod(accum * tmp, p)    
    return accum


def LCC_encoding_with_points(X, alpha_s, beta_s, p):
    # 
    #   the alpha_s is the distribution access. We evaluate p(alpha_i) fragment a finished encoding p(x) for distribution
    #   the beta_s is the encoding access: We use beta_i to build p(x) << so using beta_i to build the encoding from X  
    #      so: |beta_s| = length of the to be encoded array = z_i1,z_i2,...z_iU-T,n_iU-T+1,n_iU
    
    m, d = np.shape(X)
    U = gen_Lagrange_coeffs_lightsecagg(beta_s, alpha_s, p).astype("int64")     # U = matrix W, shape (|beta_s|, |alpha_s|)
    X_LCC = matmul_mod(U, X, p)                                                 # encoding: |beta_s| x d (one row per client)
    return X_LCC


def LCC_decoding_with_points(f_eval, eval_points, target_points, p):
    alpha_s_eval = eval_points
    beta_s = target_points
    U_dec = gen_Lagrange_coeffs_lightsecagg(beta_s, alpha_s_eval, p)   #gen_Lagrange_coeffs(beta_s, alpha_s_eval, p)
    f_recon = matmul_mod(U_dec, f_eval, p)

    return f_recon


def gen_Lagrange_coeffs_del(alpha_s, beta_s, p, is_K1=0):
    #
    #   We build the term p(X) = sum A_j prod(x-beta_i)/(beta_j-beta_i) + ... sum R_j prod(x-beta_i)/(beta_j-beta_i)
    #
    #   Note that p(beta_i) = A_i  uncloaks the coefficient (and also the R_i)
    #   We publish only p(alpha_i)
    #   Hence: 
    #
    if is_K1 == 1:
        num_alpha = 1
    else:
        num_alpha = len(alpha_s)
    U = np.zeros((num_alpha, len(beta_s)), dtype="int64")

    w = np.zeros((len(beta_s)), dtype="int64")
    for j in range(len(beta_s)):
        cur_beta = beta_s[j]
        den = PI([cur_beta - o for o in beta_s if cur_beta != o], p)
        w[j] = den

    l = np.zeros((num_alpha), dtype="int64")
    for i in range(num_alpha):
        l[i] = PI([alpha_s[i] - o for o in beta_s], p)

    for j in range(len(beta_s)):
        for i in range(num_alpha):
            den = np.mod(np.mod(alpha_s[i] - beta_s[j], p) * w[j], p)
            U[i][j] = divmod(l[i], den, p)
    return U.astype("int64")


def model_masking(weights_finite, dimensions, local_mask, prime_number):
    pos = 0
    for i, k in enumerate(weights_finite):
        tmp = weights_finite[k]
        cur_shape = tmp.shape
        d = dimensions[i]
        cur_mask = local_mask[pos : pos + d, :]
        cur_mask = np.reshape(cur_mask, cur_shape)
        weights_finite[k] += cur_mask
        weights_finite[k] = np.mod(weights_finite[k], prime_number)
        pos += d
    return weights_finite


def mask_encoding(
    total_dimension, num_clients, targeted_number_active_clients, privacy_guarantee, prime_number, local_mask
):
    # mask_encoding will for client i encode the local_mask = z_i 
    #   split this into z_ij
    #   randomly sample n_ij..
    #   multiplies this with W ?!
    
    d = total_dimension
    N = num_clients
    U = targeted_number_active_clients
    T = privacy_guarantee
    p = prime_number

    beta_s = np.array(range(N)) + 1      # the indices able to access the underlying information we encode
    alpha_s = np.array(range(U)) + (N+1)   

    """
    T = 2
    U = 5
    N=5
    d=7850"""

    # local_mask has d entries, d has to be a multiple of U-T (see padded_dimension)
    assert d % (U - T) == 0 and np.size(local_mask) == d, "mask length must be a multiple of U-T (use padded_dimension)"
    n_i = random_field_array(p, (T * d // (U - T), 1))      # randomly sample T times the blindings n_ik  \in F^( d/[U-T] )
    LCC_in = np.concatenate([np.reshape(local_mask, (d, 1)), n_i], axis=0)

    LCC_in = np.reshape(LCC_in, (U, d // (U - T)))          # rows 0..U-T-1: the U-T sub-masks [z_i]_k, rows U-T..U-1: [n_i]_k
    
    
    # LCC_in = 
    encoded_mask_set = LCC_encoding_with_points(LCC_in, alpha_s, beta_s, p).astype("int64")
    #                                                    ^^         ^^   
    #                                              distribution    encoding/decoding
    #                                              access          access
    
    return encoded_mask_set


def compute_aggregate_encoded_mask(encoded_mask_dict, p, active_clients):
    aggregate_encoded_mask = np.zeros(np.shape(encoded_mask_dict[active_clients[0]]), dtype=np.int64)
    for client_id in active_clients:
        aggregate_encoded_mask = np.mod(aggregate_encoded_mask + np.asarray(encoded_mask_dict[client_id], dtype=np.int64), p)
    return aggregate_encoded_mask


def aggregate_models_in_finite(weights_finite, prime_number):
    # 
    # Simply adding all points and taking mod to the prime number
    # 
    '''
    weights_finite : array of state_dict()
    prime_number   : size of the finite field
    '''
    w_sum = copy.deepcopy(weights_finite[0])

    for key in w_sum.keys():

        for i in range(1, len(weights_finite)):
            w_sum[key] += weights_finite[i][key]
            w_sum[key] = np.mod(w_sum[key], prime_number)

    return w_sum