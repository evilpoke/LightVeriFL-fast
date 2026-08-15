


from fastecdsa.curve import P256
import numpy as np
from utils.function import my_pk_gen, my_key_agreement, BGW_encoding, BGW_decoding, SS_decoding, SS_encoding
from utils.EC import LightVeriFL_enc_EC, LightVeriFL_dec_EC, generate_hash, \
    generate_point_EC, PI_addEC, generate_Pedersen_commitment, curve_g, curve_n
from utils.function import gen_Lagrange_coeffs, PI, divmod, gen_Lagrange_coeffs_fixed



def test_whole_01():

    N = 6
    T = int(np.floor(N / 2)) 
    U = 5
    alpha_s = np.array(range(T+1))  # np.arange(0, U)  # T+1 evaluation points for encoding inputs
    beta_s = np.arange(U, U + N)



    z = generate_point_EC()
    n_array = []  # n_array comes from the EC
    for x in range(T):
        n_array.append(generate_point_EC())


    z_tilde_array = LightVeriFL_enc_EC(z, n_array, alpha_s, beta_s, P256)
    print(z_tilde_array)
    

def test_lagrange01():

    N = 100
    T = int(np.floor(N / 2)) 
    U = 5
    alpha_s = np.array(range(T+1))  # np.arange(0, U)  # T+1 evaluation points for encoding inputs
    beta_s = np.arange(U, U + N)
    curveq = P256.q
    
    W_enc = gen_Lagrange_coeffs(alpha_s, beta_s, curveq)
    print(str(W_enc))
    
    
def test_lagrange02():

    p = 2 ** 10 - 3
    N = 100
    T = int(np.floor(N / 2)) 
    U = 5
    alpha_s = np.array(range(T+1))  # np.arange(0, U)  # T+1 evaluation points for encoding inputs
    beta_s = np.arange(U, U + N)
    curveq = P256.p
    
    W_enc = gen_Lagrange_coeffs_fixed(alpha_s, beta_s, p)
    print(str(W_enc))
    
    
test_lagrange02()
    