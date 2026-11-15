#from ....fedml_api

#from utils.function import my_pk_gen, my_key_agreement, BGW_encoding, BGW_decoding, SS_decoding, SS_encoding
#from utils.EC import LightVeriFL_enc_EC, LightVeriFL_dec_EC, generate_hash, \
#    generate_point_EC, PI_addEC, generate_Pedersen_commitment, curve_g, curve_n
#from utils.function import gen_Lagrange_coeffs, PI, divmod, gen_Lagrange_coeffs_fixed
import collections
import hashlib
import random
import binascii
import time

from tqdm import tqdm
from fedml_api.distributed.lightveriagg.utils import transform_tensor_to_finite
from fedml_api.distributed.utils.EC import add_field_elements, aggregate_el_gamal_commitments, compute_elgamal_commitment_to_hash_randomness, generate_Pedersen_commitment, generate_hash, hash_a_sample_gradient
from fedml_api.distributed.lightveriagg.LightVeriAggAggregator import aggregate_mask_functioncall_reconstruction
from fedml_api.distributed.lightveriagg.mpc_function import compute_aggregate_encoded_mask, mask_encoding, model_masking
import numpy as np
import math

p_model = 2 ** 31 - 1  # field size for model

from fastecdsa.curve import P256
from fastecdsa.point import Point
from ServerVerification.utils.EC import LightVeriFL_dec_EC, LightVeriFL_enc_EC, PI_addEC, distinct_point_compute, generate_point_EC



#xs = 0x6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296 #0xde2444bebc8d36e682edd27e0f271508617519b3221a8fa0b77cab3989da97c9
#ys = 0x4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5 #0xc093ae7ff36e5380fc01a5aad1e66659702de80f53cec576b6350b243042a256

curve_g = Point(P256.gx, P256.gy, curve=P256) #S
curve_n = P256.q


def test1_p():
    p = 2147483647
    d = int(600372.0)
    local_mask = np.random.randint(p, size=(d, 1))
    print(local_mask)


def different_generators():
    v = np.array([2,6], dtype=int)  # TODO: nothing up my sleeve
    distinct_bases = distinct_point_compute(v)
    return distinct_bases



def test2():
    system_n = curve_n
    var1 = p_model
    var2 = p_model
    var3 = p_model
    var4 = p_model
    var5 = p_model

    # el gamal thingy

    p = p_model
    #p = P256.p
    #p = 2 ** 31 - 1     # default as by init script
    total_dimension = 44
    q_bits = 10         # default as by init script

    gradients = []
    hashes = []
    randomnesses = []
    commitments = []
    masked_hashes = []
    masked_randomnesses = []
    zi_for_hashes = []
    zi_for_randomnesses = []

    for i in tqdm(range(50)): # 10 virtual clients
        
        weights = np.random.randint(var1, size=(total_dimension, 1))
        weights = {'something': weights}
        randomness = randomness = random.randrange(0, system_n) # np.random.randint(p, size=(1))
        
        gradient = transform_tensor_to_finite(weights, var2, q_bits)
        hash = hash_a_sample_gradient(total_dimension, gradient["something"])[0]
        #print(hash)
        
        hashes.append(hash)

        zi_for_hash = generate_point_EC()
        """        print("...")
        print(zi_for_hash)
        print(hash)"""

        maskedhash = zi_for_hash + hash

        zi_for_randomnes = random.randrange(0, system_n) #(system_n, size=(1))
        masked_randomnes = add_field_elements([randomness, zi_for_randomnes], system_n)  # addition under mod 
        
        gradients.append(gradient["something"])
        randomnesses.append(randomness)

        commit = compute_elgamal_commitment_to_hash_randomness(hash, randomness)

        commitments.append(commit)

        masked_hashes.append(maskedhash)
        masked_randomnesses.append(masked_randomnes)
        zi_for_randomnesses.append(zi_for_randomnes)
        zi_for_hashes.append(zi_for_hash)

    # aggregate the masks
    aggregated_mask_for_hashes = PI_addEC(zi_for_hashes)
    aggregated_mask_for_randomness = add_field_elements(zi_for_randomnesses, system_n)
    
    # aggregate the hashes
    aggregated_masked_hashed = PI_addEC(masked_hashes)

    # aggregate the randomness
    aggregated_masked_randomness = add_field_elements(masked_randomnesses, system_n)

    # demask the thingies
    #demasked_aggregated_randomness = aggregated_masked_randomness - aggregated_mask_for_randomness  #np.mod( aggregated_masked_randomness - aggregated_mask_for_randomness, p) 
    demasked_aggregated_randomness = ( aggregated_masked_randomness - aggregated_mask_for_randomness) % system_n #, p) 
    
    demasked_aggregated_hashes = aggregated_masked_hashed + (-1* aggregated_mask_for_hashes)  # 

    # aggregate the commits
    aggregated_commitments_throughopenings = compute_elgamal_commitment_to_hash_randomness(demasked_aggregated_hashes, demasked_aggregated_randomness)

    # aggregate the gradients
    agg = gradients[0]
    for grad in gradients[1:]:
        agg = agg + grad
        #agg = np.mod(agg, var4)# % p #)

    assert demasked_aggregated_hashes == PI_addEC(hashes), "err"
    assert demasked_aggregated_randomness == add_field_elements(randomnesses, system_n) , "err"

    assert hash_a_sample_gradient(total_dimension,agg ) == PI_addEC(hashes) , "err"
    aggregated_commitments_throughcommitagg = aggregate_el_gamal_commitments(commitments)
    
    assert aggregated_commitments_throughcommitagg == aggregated_commitments_throughopenings, "test"


    ##################################


    distinct_base = different_generators()
    #print(type(  randomness * distinct_base[0]))
    #exit(1)

    h1 = generate_point_EC()
    h2 = generate_point_EC()
    r1 = random.randrange(0, system_n)
    r2 = random.randrange(0, system_n)

    c1 = compute_elgamal_commitment_to_hash_randomness(h1, r1)
    c2 = compute_elgamal_commitment_to_hash_randomness(h2, r2)
    cc = aggregate_el_gamal_commitments([c1,c2])
    
    #print(cc)
    print("-------")

    hsum = PI_addEC([h1,h2])
    rsum = (r1+r2) % system_n #np.mod(r1+r2, p)
    csum = compute_elgamal_commitment_to_hash_randomness(hsum, rsum)
    
    #print(csum)

    assert csum == cc, "err"

    c2p = ()

def test1():
    # Complete runtest


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

def test9():
    # test mask reconstruction of LightSecAgg

    TODO

def test8():
    # test lightsecagg aggregation
    # no verification is happening

    d = 81      # NOTE !! the dimension needs to fit to d // (U - T))
    weights_field = p_model
    #somerandomkeysize = 2^32-1
    alpha = np.array(random.sample(range(1, d + 1), k=d))  
    
    system_n = curve_n
    q_bits = 10
    N = 20
    numberclients = N
    
    T = int(np.floor(N / 2)-1) 
    U = 18
    idx_clients_that_can_aggregate = random.sample(range(N), k=U)  # NOTE: this will only accept at most k=U and not more in aggregate_mask_functioncall_reconstruction
    
    alpha_s = np.array(range(T+1))  # np.arange(0, U)  # T+1 evaluation points for encoding inputs
    beta_s = np.arange(U, U + N)

    internal_z_tilde_array = [0]*N   # fragmented z masks for hash

    send_to_artifical_target_second = dict()   # gradient
    for i in range(N):
        send_to_artifical_target_second[i] = []
    
    to_server_to_aggregate = []
    whitebox_zs = []

    gradients = []
    zi_for_gradient = []
    print("Generating stuff...")
    stored_zs = []
    stored_tilde_zs = []
    stored_commitments = []
    submitted_masked_gradients = []
    submitted_aggregated_z = []
    
    for i in tqdm(range(numberclients)):
        
        # handling mask for gradient
        z = np.random.randint(p_model, size=(d, 1)) 
        zi_for_gradient.append(z)
        encoded_mask_set = mask_encoding(d, N, U, T, p_model, z)
        for artificial_target in range(N):
            send_to_artifical_target_second[artificial_target].append(encoded_mask_set[artificial_target])
        
    for i in tqdm(range(numberclients)): 
        # io
        my_inbox_2 = send_to_artifical_target_second[i]
        stored_zs.append(my_inbox_2)

    for i in tqdm(range(numberclients)): 
        # ----- training, commitment  -----
        # ----- creating a commitment -----
        weights = np.random.randint(99, size=(d, 1)) #np.random.randint(weights_field, size=(d, 1))
        weights = {'something': weights}
        gradient = transform_tensor_to_finite(weights, weights_field, q_bits)
        gradients.append(gradient)

    for i in tqdm(range(numberclients)): 
        # submitting mask model and masked verification material

        z = zi_for_gradient[i]
        gradient = gradients[i]
        
        masked_weights = model_masking(gradient, [d], z, p_model)
        submitted_masked_gradients.append(masked_weights)

    aggregate_encoded_mask_dict = dict()

    for i in tqdm( range(N)): 
        # aggregate masks

        active_clients_first_round = [i for i in range(N)]
        aggregatedshards = compute_aggregate_encoded_mask(send_to_artifical_target_second[i], p_model, active_clients_first_round)    

        aggregate_encoded_mask_dict[i] = aggregatedshards

        
    # server is solving the aggregated mask 
    
    # --- model ----
    # basically def aggregate_model_reconstruction(...
    decoded_z_sum = aggregate_mask_functioncall_reconstruction(d,N,U,T,p_model,idx_clients_that_can_aggregate, aggregate_encoded_mask_dict)

    
    # aggreate masked gradients and subtract mask
    averaged_params = submitted_masked_gradients[idx_clients_that_can_aggregate[0]] 
    pos = 0
    for j, k in enumerate(averaged_params):
        for i, client_idx in enumerate(idx_clients_that_can_aggregate):
            local_model_params = submitted_masked_gradients[client_idx]
            if i == 0:
                averaged_params[k] = local_model_params[k]
            else:
                averaged_params[k] += local_model_params[k]
        cur_shape = np.shape(averaged_params[k])
        d = d #self.dimensions[j]
        cur_mask = np.array(decoded_z_sum[pos : pos + d, :])
        cur_mask = np.reshape(cur_mask, cur_shape)

        # Cancel out the aggregate-mask to recover the aggregate-model
        averaged_params[k] -= cur_mask
        pos += d

    
    grad = gradients[0]["something"]
    for g in range(1,N):
        
        
        grad += gradients[g]["something"]


    # averaged_params is the new model
    #print("--------------")
    # decoded_aggregated_hashes and decoded_aggregated_randomness are decoded stuff
    #print(averaged_params["something"])
    #print(grad)
    print("averaged_params[something]: " + str(averaged_params["something"]))
    print("grads: " + str(grad))
    
    assert averaged_params["something"][0] == grad[0]


def test5():
    # complete runtest with pederson "variant"
    # - simplified setup with only checking the surviving users' hash and not also the dropped users hash
    # - 
    #

    d = 21
    weights_field = p_model
    somerandomkeysize = 2^32-1
    alpha = np.array(random.sample(range(1, d + 1), k=d))  
    
    system_n = curve_n
    q_bits = 10
    N = 20
    numberclients = N
    Pedersen_coeff = np.array(random.sample(range(1, somerandomkeysize), 2))  # distributed by server, # we can use curve.n instead of 100 here. -> we did now


    T = int(np.floor(N / 2)) 
    U = 18
    idx_clients_that_can_aggregate = random.sample(range(N), k=U)
    
    
    alpha_s = np.array(range(T+1))  # np.arange(0, U)  # T+1 evaluation points for encoding inputs
    beta_s = np.arange(U, U + N)

    internal_z_tilde_array = [0]*N   # fragmented z masks for hash

    send_to_artifical_target = dict() #[0]*N            # randomness and hash
    for i in range(N):
        send_to_artifical_target[i] = []

    send_to_artifical_target_second = dict()   # gradient
    for i in range(N):
        send_to_artifical_target_second[i] = []
    
    to_server_to_aggregate = []
    whitebox_zs = []


    gradients = []
    hashes = []
    randomnesses = []
    commitments = []
    masked_hashes = dict()
    masked_randomnesses = dict()
    zi_for_hashes_and_randomness = []
    zi_for_gradient = []
    print("Generating stuff...")
    stored_zs = []
    stored_tilde_zs = []
    stored_commitments = []
    submitted_masked_gradients = dict()
    submitted_aggregated_zv = dict()
    submitted_aggregated_z = dict()
    
    for i in tqdm(range(numberclients)):

        # handling mask for verification
        z = generate_point_EC()
        zi_for_hashes_and_randomness.append(z)

        n_array = []  # n_array comes from the EC
        for x in range(T):
            n_array.append(generate_point_EC())
        z_tilde_array = LightVeriFL_enc_EC(z, n_array, alpha_s, beta_s, P256)
        for artificial_target in range(N):
            send_to_artifical_target[artificial_target].append(z_tilde_array[artificial_target])
        
        # handling mask for gradient
        z = np.random.randint(p_model, size=(d, 1)) 
        zi_for_gradient.append(z)
        encoded_mask_set = mask_encoding(d, N, U, T, p_model, z)
        for artificial_target in range(N):
            send_to_artifical_target_second[artificial_target].append(encoded_mask_set[artificial_target])

    for i in tqdm(range(numberclients)): 
        # io
        my_inbox_1 = send_to_artifical_target[i]
        my_inbox_2 = send_to_artifical_target_second[i]
        stored_zs.append(my_inbox_2)
        stored_tilde_zs.append(my_inbox_1)

    for i in tqdm(range(numberclients)): 
        # ----- training, commitment  -----
        # ----- creating a commitment -----
        weights = np.random.randint(weights_field, size=(d, 1))
        weights = {'something': weights}
        gradient = transform_tensor_to_finite(weights, weights_field, q_bits)
        gradients.append(gradient)
        hash = generate_hash(gradient, alpha, d)  # should have no differce to: #hash_a_sample_gradient(total_dimension, gradient["something"])[0]

        
        Pedersen_noise = generate_point_EC()
        z = zi_for_hashes_and_randomness[i] # simply in storage...
        
        Pedersen_g = (Pedersen_coeff[0] * curve_g)
        Pedersen_l = (Pedersen_coeff[1] * curve_g)
        commitment = generate_Pedersen_commitment(hash, Pedersen_g, Pedersen_l, Pedersen_noise)

        commitments.append(commitment)
        hashes.append(hash)
        randomnesses.append(Pedersen_noise)


    for i in tqdm(range(numberclients)): 
        # submitting mask model and masked verification material

        z_tilde = zi_for_hashes_and_randomness[i]
        z = zi_for_gradient[i]
        gradient = gradients[i]


        if i in idx_clients_that_can_aggregate:
            h = hashes[i]
            masked_hash = h + z_tilde
            masked_hashes[i] = masked_hash

            Pedersen_noise = randomnesses[i]
            masked_noise = (Pedersen_noise + z)
            masked_randomnesses[i] = masked_noise

            masked_weights = model_masking(gradient, d, z, p_model)
            submitted_masked_gradients[i] = masked_weights

    preceived_set_of_survivers = set(masked_hashes.keys()).intersection(set(masked_randomnesses.keys())).intersection(set(submitted_masked_gradients.keys()))

    assert preceived_set_of_survivers == set(idx_clients_that_can_aggregate), "Dropouts are not happening as predicted"
    # every client now has idx_clients_that_can_aggregate

    for i in tqdm( random.sample(range(N), k=N-1)): 
        # aggregate masks

        aggregated_mask_for_verification =  PI_addEC(stored_tilde_zs[i][idx_clients_that_can_aggregate])

        encoded_mask_dict = stored_zs[i]
        aggregated_mask_for_model = compute_aggregate_encoded_mask(encoded_mask_dict, p_model, idx_clients_that_can_aggregate)

        submitted_aggregated_zv[i] = aggregated_mask_for_verification
        submitted_aggregated_z[i] = aggregated_mask_for_model

    # server is solving the aggregated mask and aggregated hash

    # --- hash and randomness ---
    decoded_zv_sum = LightVeriFL_dec_EC(submitted_aggregated_zv.values(), alpha_s, beta_s[idx_clients_that_can_aggregate], P256)
    minus_decoded_zv_sum = (-1 * decoded_zv_sum)
    decoded_aggregated_hashes = PI_addEC(masked_hashes.values()) + minus_decoded_zv_sum
    decoded_aggregated_randomness = PI_addEC(masked_randomnesses.values()) + minus_decoded_zv_sum

    # --- model ----
    # basically def aggregate_model_reconstruction(...
    decoded_z_sum = aggregate_mask_functioncall_reconstruction(d,N,U,T,p_model,idx_clients_that_can_aggregate, idx_clients_that_can_aggregate)
    averaged_params = submitted_masked_gradients[idx_clients_that_can_aggregate[0]]  # TODO << maybe put [] / dict around?
    pos = 0
    for j, k in enumerate(averaged_params): # NOTE: this loop will only execute once!. So we are sticking as closely as possible to the original function
        for i, client_idx in enumerate(idx_clients_that_can_aggregate):
            local_model_params = submitted_masked_gradients[client_idx]
            if i == 0:
                averaged_params[k] = local_model_params[k]
            else:
                averaged_params[k] += local_model_params[k]
        cur_shape = np.shape(averaged_params[k])
        d = d #self.dimensions[j]
        cur_mask = np.array(decoded_z_sum[pos : pos + d, :])
        cur_mask = np.reshape(cur_mask, cur_shape)

        # Cancel out the aggregate-mask to recover the aggregate-model
        averaged_params[k] -= cur_mask
        pos += d

    # averaged_params is the new model
    # decoded_aggregated_hashes and decoded_aggregated_randomness are decoded stuff

    for i in tqdm(idx_clients_that_can_aggregate):
        # each surviving client checks

        self_constructed_agg_commit = generate_Pedersen_commitment(decoded_aggregated_hashes, Pedersen_g, Pedersen_l, decoded_aggregated_randomness)
        commitments_summed = PI_addEC(commitments[idx_clients_that_can_aggregate])
        commitments_summed = (commitments_summed + (-(N - 1) * (Pedersen_g + Pedersen_l)))  # TODO << why the fuck ?? test this!!

        if self_constructed_agg_commit != commitments_summed:
            raise ValueError("commit not correct")

        if generate_hash(averaged_params, alpha, d) != PI_addEC(hashes[idx_clients_that_can_aggregate]):
            raise ValueError("hashes not correct")

def test7():
    # testing if the commit(=el gamal)/hash test works
    randomness_field = curve_n
    N = 20
    
    commitments = []
    hashes = []
    randomnesses = []

    for i in tqdm(range(N)):

        hash = generate_point_EC()
        randomness = random.randrange(1, randomness_field)
        commit = compute_elgamal_commitment_to_hash_randomness(hash, randomness)
        
        randomnesses.append(randomness)
        commitments.append(commit)
        hashes.append(hash)
    
    # checks


    hashes_summed_up = PI_addEC(hashes)
    randomnesses_summed_up = add_field_elements(randomnesses,curve_n)

    self_constructed_agg_commit = compute_elgamal_commitment_to_hash_randomness(hashes_summed_up,randomnesses_summed_up)

    commitments_summed = aggregate_el_gamal_commitments(commitments)
    

    if self_constructed_agg_commit != commitments_summed:
        raise ValueError("commit not correct")



def test3():

    # LightVeriFL with 
    #   - el gamal commit, hash masked with z, sharded and reconstructed, but not the randomness mask. The randomness masks are given 
    #       - furthermore, nicht alle aggregieren. Und zudem wird nicht über alle aggregiert sondern nur über die offiziellen surviver
    #   - random gradient, but the gradient is not masked.


    weights_field = p_model
    randomness_field = curve_n #2**255
    randomness_field_mod_cutoff = curve_n

    total_dimension = 21
    system_n = curve_n
    q_bits = 10
    N = 20
    numberclients = N


    T = int(np.floor(N / 2)) 
    U = 14
    alpha_s = np.array(range(T+1))  # np.arange(0, U)  # T+1 evaluation points for encoding inputs
    beta_s = np.arange(U, U + N)

    internal_z_tilde_array = [0]*N   # fragmented z masks for hash
    internal_z_tilde_array_rnd = [0]*N   # fragmented z masks for randomness

    # postoffice
    send_to_artifical_target_sharehashmask = dict() #[0]*N
    for i in range(N):
        send_to_artifical_target_sharehashmask[i] = []
    send_to_artifical_target_sharerndmask = dict() #[0]*N
    for i in range(N):
        send_to_artifical_target_sharerndmask[i] = []

    
    to_server_to_aggregate_hashmask = []
    to_server_to_aggregate_rndmask = []
    whitebox_zs = []

    modelsubmittingset = (1,1+U)
    gradients = []
    hashes = []
    randomnesses = []
    commitments = []
    masked_hashes = []
    masked_randomnesses = []
    zi_for_hashes = []
    zi_for_randomnesses = []
    zi_for_gradient = []
    print("Generating stuff...")
    for i in tqdm(range(numberclients)): 
        weights = np.random.randint(weights_field, size=(total_dimension, 1))
        weights = {'something': weights}
        gradient = transform_tensor_to_finite(weights, weights_field, q_bits)
        hash = hash_a_sample_gradient(total_dimension, gradient["something"])[0]

        randomness = randomness = random.randrange(1, randomness_field) # np.random.randint(p, size=(1))
        
        hashes.append(hash)

        zi_for_hash = generate_point_EC()
        n_array = []  # n_array comes from the EC
        for x in range(T):
            n_array.append(generate_point_EC())

        # sharing the mask for the hash
        internal_z_tilde_array[i] = LightVeriFL_enc_EC(zi_for_hash, n_array, alpha_s, beta_s, P256)
        
        maskedhash = zi_for_hash + hash

        zi_for_randomnes = random.randrange(1, randomness_field) #(system_n, size=(1))
        masked_randomnes = add_field_elements([randomness, zi_for_randomnes], randomness_field)  # addition under mod 

        # not sharding the mask for the randomness
        
        
        gradients.append(gradient["something"])
        randomnesses.append(randomness)

        commit = compute_elgamal_commitment_to_hash_randomness(hash, randomness)

        commitments.append(commit)

        # official protocol
        masked_hashes.append(maskedhash)
        masked_randomnesses.append(masked_randomnes)

        # sanity check debug
        zi_for_randomnesses.append(zi_for_randomnes)
        zi_for_hashes.append(zi_for_hash)


    print("Sending...")
    for i in tqdm(range(numberclients)):
        for artificial_target in range(N):
            
            # z mask for hash
            tx_data = internal_z_tilde_array[i] 
            send_to_artifical_target_sharehashmask[artificial_target].append(tx_data[artificial_target])

            # z mask for randomness
            #tx_data = internal_z_tilde_array_rnd[i] 
            #send_to_artifical_target_sharerndmask[artificial_target].append(tx_data[artificial_target])


    print("Some clients now aggregate...")
    for receiver in range( modelsubmittingset[0] , modelsubmittingset[1]-1):  # nicht alle reagieren

        ### --- aggregate z mask for hash ---

        # gather all hash masks
        all_packages_to_reciever = send_to_artifical_target_sharehashmask[receiver][modelsubmittingset[0]:modelsubmittingset[1]]  # und vom fed wurde bekannt gegeben, dass 1:1+U surviver sind
        
        # aggregate encoding masks to create the one mask
        z_tilde_mul_surviving = PI_addEC(all_packages_to_reciever)

        # send
        to_server_to_aggregate_hashmask.append(z_tilde_mul_surviving)

        ### --- aggregate z mask for randomness ---
        
        # gather all randomness masks
        # Debug: all_packages_to_reciever = send_to_artifical_target_sharerndmask[receiver][modelsubmittingset[0]:modelsubmittingset[1]]  # und vom fed wurde bekannt gegeben, dass 1:1+U surviver sind

        # aggregate encoding masks to create the one mask
        #Debug: z_tilde_mul_surviving = add_field_elements(all_packages_to_reciever, system_n)
        
        # send
        #Debug: to_server_to_aggregate_rndmask.append(z_tilde_mul_surviving)


    # decode the masks for the hash; simply gather the masks for the randomness
    aggregated_mask_for_hash = LightVeriFL_dec_EC(to_server_to_aggregate_hashmask, alpha_s, beta_s[modelsubmittingset[0]:modelsubmittingset[1]-1],P256)
    
    assert aggregated_mask_for_hash == PI_addEC(zi_for_hashes[modelsubmittingset[0]:modelsubmittingset[1]]), "fail"

    # cheating here (debug)
    aggregated_mask_for_randomness = add_field_elements(zi_for_randomnesses[modelsubmittingset[0]:modelsubmittingset[1]], system_n)
    
    # aggregate the hashes
    aggregated_masked_hashes = PI_addEC(masked_hashes[modelsubmittingset[0]:modelsubmittingset[1]])

    # aggregate the randomness
    aggregated_masked_randomness = add_field_elements(masked_randomnesses[modelsubmittingset[0]:modelsubmittingset[1]], system_n)

    # demask the thingies
    demasked_aggregated_randomness = aggregated_masked_randomness - aggregated_mask_for_randomness  #np.mod( aggregated_masked_randomness - aggregated_mask_for_randomness, p) 
    demasked_aggregated_randomness = ( aggregated_masked_randomness - aggregated_mask_for_randomness) % system_n #, p) 
    
    demasked_aggregated_hashes = aggregated_masked_hashes + (-1* aggregated_mask_for_hash)  # 

    # aggregate the commits
    aggregated_commitments_throughopenings = compute_elgamal_commitment_to_hash_randomness(demasked_aggregated_hashes, demasked_aggregated_randomness)

    assert aggregated_commitments_throughopenings == aggregate_el_gamal_commitments(commitments[modelsubmittingset[0]:modelsubmittingset[1]])

    print("ok")

    

def test4():
    # LightVeriFL just exchanged with el gamal commit

    #system_n = p_model
    #for (var1,var2,var3,var4,var5,var5) in tuple(itertools.product(*setss))
    weights_field = p_model
    randomness_field = curve_n #2**255
    randomness_field_mod_cutoff = curve_n
    
    var2 = p_model
    var3 = p_model
    var4 = p_model
    var5 = p_model

    # el gamal thingy

    p = p_model
    #p = P256.p
    #p = 2 ** 31 - 1     # default as by init script
    total_dimension = 44
    q_bits = 10         # default as by init script

    gradients = []
    hashes = []
    randomnesses = []
    commitments = []
    masked_hashes = []
    masked_randomnesses = []
    zi_for_hashes = []
    zi_for_randomnesses = []

    for i in tqdm(range(50)): # 10 virtual clients
        
        weights = np.random.randint(weights_field, size=(total_dimension, 1))
        weights = {'something': weights}
        randomness = random.randrange(1, randomness_field) # np.random.randint(p, size=(1))
        
        gradient = transform_tensor_to_finite(weights, weights_field, q_bits)
        hash = hash_a_sample_gradient(total_dimension, gradient["something"])[0]
        #print(hash)
        
        hashes.append(hash)

        zi_for_hash = generate_point_EC()

        """        print("...")
        print(zi_for_hash)
        print(hash)"""


        maskedhash = zi_for_hash + hash

        zi_for_randomnes = random.randrange(1, randomness_field) #(system_n, size=(1))
        masked_randomnes = add_field_elements([randomness, zi_for_randomnes], randomness_field_mod_cutoff)  # addition under mod 
        
        gradients.append(gradient["something"])
        randomnesses.append(randomness)

        commit = compute_elgamal_commitment_to_hash_randomness(hash, randomness)

        commitments.append(commit)

        masked_hashes.append(maskedhash)
        masked_randomnesses.append(masked_randomnes)
        zi_for_randomnesses.append(zi_for_randomnes)
        zi_for_hashes.append(zi_for_hash)

    # aggregate the masks
    aggregated_mask_for_hashes = PI_addEC(zi_for_hashes)
    aggregated_mask_for_randomness = add_field_elements(zi_for_randomnesses, randomness_field_mod_cutoff)
    
    # aggregate the hashes
    aggregated_masked_hashed = PI_addEC(masked_hashes)

    # aggregate the randomness
    aggregated_masked_randomness = add_field_elements(masked_randomnesses, randomness_field_mod_cutoff)

    # demask the thingies
    #demasked_aggregated_randomness = aggregated_masked_randomness - aggregated_mask_for_randomness  #np.mod( aggregated_masked_randomness - aggregated_mask_for_randomness, p) 
    demasked_aggregated_randomness = ( aggregated_masked_randomness - aggregated_mask_for_randomness) % randomness_field_mod_cutoff #, p) 
    
    demasked_aggregated_hashes = aggregated_masked_hashed + (-1* aggregated_mask_for_hashes)  # 

    # aggregate the commits
    aggregated_commitments_throughopenings = compute_elgamal_commitment_to_hash_randomness(demasked_aggregated_hashes, demasked_aggregated_randomness)

    # aggregate the gradients
    agg = gradients[0]
    for grad in gradients[1:]:
        agg = agg + grad
        #agg = np.mod(agg, var4)# % p #)

    assert demasked_aggregated_hashes == PI_addEC(hashes), "err"
    assert demasked_aggregated_randomness == add_field_elements(randomnesses, randomness_field) , "err"

    assert hash_a_sample_gradient(total_dimension,agg ) == PI_addEC(hashes) , "err"
    aggregated_commitments_throughcommitagg = aggregate_el_gamal_commitments(commitments)

    assert aggregated_commitments_throughcommitagg == aggregated_commitments_throughopenings, "test"


    ##################################


    distinct_base = different_generators()
    #print(type(  randomness * distinct_base[0]))
    #exit(1)

    h1 = generate_point_EC()
    h2 = generate_point_EC()
    r1 = random.randrange(1, randomness_field)
    r2 = random.randrange(1, randomness_field)

    c1 = compute_elgamal_commitment_to_hash_randomness(h1, r1)
    c2 = compute_elgamal_commitment_to_hash_randomness(h2, r2)
    cc = aggregate_el_gamal_commitments([c1,c2])
    
    #print(cc)
    print("-------")

    hsum = PI_addEC([h1,h2])
    rsum = (r1+r2) % randomness_field #np.mod(r1+r2, p)
    csum = compute_elgamal_commitment_to_hash_randomness(hsum, rsum)
    
    #print(csum)

    assert csum == cc, "err"

    c2p = ()



if __name__ == "__main__":
    #print("=======Test 2===================")

    #test2()     # works
    #test1_p()  works

    #print("=======Test 1===================")

    #test1()     # works

    #test3()     # works
    # test4()  # works
    
    #test7()    works

    
    #test5()

    test8()
    
    #for i in range(22):
    #    test4()






