


from fastecdsa.curve import P256
import numpy as np
from utils.function import my_pk_gen, my_key_agreement, BGW_encoding, BGW_decoding, SS_decoding, SS_encoding
from utils.EC import LightVeriFL_enc_EC, LightVeriFL_dec_EC, generate_hash, \
    generate_point_EC, PI_addEC, generate_Pedersen_commitment, curve_g, curve_n
from utils.function import gen_Lagrange_coeffs, PI, divmod, gen_Lagrange_coeffs_fixed



def test_whole_01():
    """
    Splitting and merging again


    """

    N = 20
    T = int(np.floor(N / 2)) 
    U = 14
    alpha_s = np.array(range(T+1))  # np.arange(0, U)  # T+1 evaluation points for encoding inputs
    beta_s = np.arange(U, U + N)

    internal_z_tilde_array = [0]*N
    send_to_artifical_target = dict() #[0]*N
    for i in range(N):
        send_to_artifical_target[i] = []
    
    to_server_to_aggregate = []
    whitebox_zs = []

    for sender in range(N):
        z = generate_point_EC()
        whitebox_zs.append(z)
        n_array = []  # n_array comes from the EC
        for x in range(T):
            n_array.append(generate_point_EC())

        internal_z_tilde_array[sender] = LightVeriFL_enc_EC(z, n_array, alpha_s, beta_s, P256)

    #print(z_tilde_array)
    for sender in range(N):
        for artificial_target in range(N):
            tx_data = internal_z_tilde_array[sender]
            send_to_artifical_target[artificial_target].append(tx_data[artificial_target])

    # we are now to only select these survivings:
    #1:1+14

    for receiver in range(2,1+U):
        all_packages_to_reciever = send_to_artifical_target[receiver][1:1+U]

        # aggregate encoding masks to create the one mask
        z_tilde_mul_surviving = PI_addEC(all_packages_to_reciever)

        to_server_to_aggregate.append(z_tilde_mul_surviving)

    # server now decodes to get \sum z_i

    dec_z = LightVeriFL_dec_EC(to_server_to_aggregate, alpha_s, beta_s[2:1+U], P256)
    print(dec_z)
    acc = 0

    #for i in range(1,1+U):
    acc = PI_addEC(whitebox_zs[1:1+U])#[acc,)     + [i]

    print(acc)

    assert acc == dec_z , "Fail"

    print("done")
    






def test_lagrange01():

    N = 100
    T = int(np.floor(N / 2)) 
    U = 51
    alpha_s = list(range(T+1)) #np.array(range(T+1))  # np.arange(0, U)  # T+1 evaluation points for encoding inputs
    beta_s = list(range(U, U + N))  # np.arange(U, U + N)
    curveq = P256.q
    
    W_enc = gen_Lagrange_coeffs(alpha_s, beta_s, curveq)
    #print(str(W_enc))
    
    
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
    
test_whole_01()
    