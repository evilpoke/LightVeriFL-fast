import logging
import os
import copy
import pickle
import random
import numpy as np
import torch


def save_model_dict_to_pickle(path, weights):
    with open(path, "wb") as fh:
        pickle.dump(weights, fh)

def load_model_dict_from_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def finite_to_signed(X_q, p):
    """Centered lift of field elements of F_p: [0, p) -> (-(p-1)/2, (p-1)/2] (int64).

    This is the integer a quantized value encodes (same convention as my_q_inv). The linearly
    homomorphic EC hash is applied to these integers, so that h(y) = sum_i h(x_i) holds for the
    aggregate y = sum_i x_i mod p as long as the real-valued aggregate does not overflow the field.
    """
    X_q = np.asarray(X_q, dtype=np.int64)
    return np.where(X_q > (p - 1) // 2, X_q - p, X_q)


def flatten_finite_model(model_params, p):
    """Model dict in F_p (state_dict key order) -> one signed int64 vector of length d."""
    return np.concatenate([finite_to_signed(np.asarray(model_params[k]).reshape(-1), p) for k in model_params.keys()])


def finite_aggregate_to_averaged_tensor(aggregate_finite, p, q_bits, num_aggregated):
    """sum_i x_i in F_p  ->  real-valued average model (dict of float tensors)."""
    out = dict()
    for k in aggregate_finite.keys():
        tmp_real = my_q_inv(np.asarray(aggregate_finite[k], dtype=np.int64), q_bits, p) / float(num_aggregated)
        out[k] = torch.as_tensor(np.asarray(tmp_real), dtype=torch.float32)
    return out


def model_dimension(weights):
    logging.info("Get model dimension")
    dimensions = []
    for k in weights.keys():
        tmp = weights[k].cpu().detach().numpy()
        cur_shape = tmp.shape
        _d = int(np.prod(cur_shape))
        dimensions.append(_d)
    total_dimension = sum(dimensions)
    logging.info("Dimension of model d is %d." % total_dimension)
    return dimensions, total_dimension


def transform_tensor_to_finite(model_params, p, q_bits):
    for k in model_params.keys():
        v = model_params[k]
        tmp = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.array(v)
        tmp_finite = my_q(tmp, q_bits, p)
        model_params[k] = tmp_finite
    return model_params


def transform_finite_to_tensor(model_params, p, q_bits):
    for k in model_params.keys():
        tmp = np.array(model_params[k])
        tmp_real = my_q_inv(tmp, q_bits, p)
        """
        Handle two types of numpy variables, array and float
        0 - Wed, 13 Oct 2021 07:50:59 utils.py[line:33] 
        DEBUG tmp_real = [4.20669909e+07 3.95378213e+08 3.20326493e+08 2.69357568e+08
         1.50241777e+08 4.17396229e+08 4.70202170e+07 2.97959914e+08
         3.03893524e+08 5.43255322e+07 4.22816876e+08 3.57158791e+06
         1.56766933e+08 2.36449153e+08 6.66361822e+07 1.66616215e+08]
        0 - Wed, 13 Oct 2021 07:50:59 utils.py[line:33] DEBUG tmp_real = 256812209.4375
        """
        # logging.debug("tmp_real = {}".format(tmp_real))
        tmp_real = torch.Tensor([tmp_real]) if isinstance(tmp_real, np.floating) else torch.Tensor(tmp_real)
        model_params[k] = tmp_real
    return model_params


def transform_list_to_tensor(model_params_list):
    for k in model_params_list.keys():
        model_params_list[k] = torch.from_numpy(np.asarray(model_params_list[k])).float()
    return model_params_list


def transform_tensor_to_list(model_params):
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params


def post_complete_message_to_sweep_process(args):
    pipe_path = "./tmp/fedml"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, "w") as pipe:
        pipe.write("training is finished! \n%s\n" % (str(args)))


def modular_inv(a, p):
    # exact Python-int arithmetic: works for the small model field (2^31-1) as well as for the
    # 256-bit group order of P-256 (numpy int64 overflowed there: "Python int too large to convert to C long")
    return pow(int(a) % int(p), -1, int(p))


def divmod(_num, _den, _p):
    # compute num / den modulo prime p
    return (int(_num) % int(_p)) * modular_inv(_den, _p) % int(_p)


def PI(vals, p):  # upper-case PI -- product of inputs
    accum = 1
    for v in vals:
        accum = (accum * (int(v) % int(p))) % int(p)
    return accum


def LCC_encoding_with_points(X, alpha_s, beta_s, p):
    m, d = np.shape(X)
    U = gen_Lagrange_coeffs(beta_s, alpha_s, p).astype("int64")
    X_LCC = np.zeros((len(beta_s), d), dtype="int64")
    for i in range(len(beta_s)):
        X_LCC[i, :] = np.dot(np.reshape(U[i, :], (1, len(alpha_s))), X)
    return np.mod(X_LCC, p)


def LCC_decoding_with_points(f_eval, eval_points, target_points, p):
    alpha_s_eval = eval_points
    beta_s = target_points
    U_dec = gen_Lagrange_coeffs(beta_s, alpha_s_eval, p)
    f_recon = np.mod((U_dec).dot(f_eval), p)

    return f_recon

def gen_Lagrange_coeffs_lightsecagg(alpha_s, beta_s, p, is_K1=0):
    """
    Yet another gen_lagrange_coeff implementation
    """
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


def gen_Lagrange_coeffs(evalpoints_in, evalpoints_out, p, is_K1=0):
    '''
    LIGHTVERIFL implementation
    input
        - evalpoints_in  : array of evaluation points corresponding to encoding inputs (K)
        - evalpoints_out : array of evaluation points corresponding to encoding outputs (N)
        - p : finite field size
        - is_K1 : True when K=1

    output
        - U : matrix of lagrange coefficients (K x N)
    '''
    #print("DEBUG: evals = "+str(evalpoints_in)+"")
    rows, cols = (len(evalpoints_out), len(evalpoints_in))
    U = [[[] for i in range(cols)] for i in range(rows)]  # [[0] * cols] * rows
    # U = np.zeros((len(evalpoints_out), len(evalpoints_in)), dtype='int64')

    w = []  # np.zeros((len(evalpoints_in)), dtype='int64')
    for j in range(len(evalpoints_in)):
        cur_beta = evalpoints_in[j]
        #try:
        den = PI([cur_beta - o for o in evalpoints_in if cur_beta != o], p)
        #except Exception as e:
        #    print("DEBUG: input to PI = "+str([cur_beta - o for o in evalpoints_in if cur_beta != o]))
        w.append(den)  # w[j] = den

    l = []  # np.zeros((len(evalpoints_out)),dtype='int64')
    for i in range(len(evalpoints_out)):
        #tmp = int(PI([evalpoints_out[i] - o for o in evalpoints_in], p)) % p
        #l.append(tmp)
        l.append(PI([evalpoints_out[i] - o for o in evalpoints_in], p))
        
        # l[i] = PI([evalpoints_out[i] - o for o in evalpoints_in], p)

    for j in range(len(evalpoints_in)):
        for i in range(len(evalpoints_out)):
            den =  ((int(evalpoints_out[i]) - int(evalpoints_in[j])) % p) * w[j] % p # np.mod(np.mod(evalpoints_out[i] - evalpoints_in[j], p) * w[j], p)
            #try:
            U[i][j] = divmod(l[i], den, p)
            #except Exception as e:
            #    print("DEBUG: l[i] = "+str(l[i])+" den="+str(den)+" p="+str(p))

    return U  # U.astype('int64')


# supposed gen_lagrange from lightsecagg
def gen_Lagrange_coeffs_del(alpha_s, beta_s, p, is_K1=0):
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


def my_q(X, q_bit, p):
    # quantisation with parameter q_bit and prime p

    
    # float64: in float32 (torch weights) p = 2^31-1 is not representable and every negative value was
    # quantized to ~2^31 (garbage), i.e. the models lost their negative weights
    X_int = np.round(np.asarray(X, dtype=np.float64) * (2 ** q_bit))


    # \phi(x)
    is_negative = (abs(np.sign(X_int)) - np.sign(X_int)) / 2
    out = X_int + p * is_negative
    return out.astype("int64")


def my_q_inv(X_q, q_bit, p):
    # "interpolation"?!?! to 
    flag = X_q - (p - 1) / 2
    is_negative = (abs(np.sign(flag)) + np.sign(flag)) / 2
    X_q = X_q - p * is_negative
    return X_q.astype(float) / (2 ** q_bit)


