#from ....fedml_api

import numpy as np



def test1():
    p = 2147483647
    d = int(600372.0)
    local_mask = np.random.randint(p, size=(d, 1))
    print(local_mask)


def different_generators():
    v = np.array([2,6], dtype=int)  # TODO: nothing up my sleeve
    distinct_bases = distinct_point_compute(v)
    return distinct_bases


def hash_a_sample_gradient(total_dimension, gradient):
    
    alpha = np.ones((total_dimension,), dtype=int)

    distinct_bases = distinct_point_compute(alpha)
    # compute g_i ** gradient[i] for a client
    temp_hash = tuple(gradient[i] * distinct_bases[i] for i in range(total_dimension))
    # compute the product for [d]. This product translates into addition on the elliptic curve
    hash_client = temp_hash[0]
    for i in range(1, total_dimension):
        hash_client = hash_client + temp_hash[i]

    return hash_client

def compute_elgamal_commitment_to_hash_randomness(hash, randomness):
    # hash is a group element
    # randomness is a field / scalar element
    # returns the tuple (c1, c2)

    #alpha = np.ones((1,), dtype=int)

    distinct_base = different_generators()


    g_powof_r = randomness * distinct_base[0]
    h_powof_r = randomness * distinct_base[1]
    c1 = g_powof_r

    M_with_hash = [hash, h_powof_r]
    c2 = PI_addEC(M_with_hash)

    return (c1, c2)


def test2():

    # el gamal thingy

    p = 2 ** 31 - 1     # default as by init script
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

    for i in range(10): # 10 virtual clients
        
        weights = np.random.randint(p, size=(total_dimension, 1))
        randomness = np.random.randint(p, size=(1))

        gradient = transform_tensor_to_finite(weights, p, q_bits)
        hash = hash_a_sample_gradient(gradient)

        hashes.append(hash)

        zi_for_hash = generate_point_EC()
        maskedhash = zi_for_hash + hash

        zi_for_randomnes = np.random.randint(p, size=(1))
        masked_randomnes = model_masking(randomness, zi_for_randomnes, 1, p)  # addition under mod 
        
        gradients.append(gradient)
        randomnesses.append(randomnesses)

        commit = compute_elgamal_commitment_to_hash_randomness(hash, randomness)

        commitments.append(commit)

        masked_hashes.append(maskedhash)
        masked_randomnesses.append(masked_randomnes)
        zi_for_randomnesses.append(zi_for_randomnes)
        zi_for_hashes.append(zi_for_hash)

    # aggregate the masks
    aggregated_mask_for_hashes = PI_addEC(zi_for_hashes)
    aggregated_mask_for_randomness = add_field_elements(zi_for_randomnesses)
    
    # aggregate the hashes
    aggregated_masked_hashed = PI_addEC(masked_hashes)

    # aggregate the randomness    
    aggregated_masked_randomness = add_field_elements(masked_randomnesses)

    # demask the thingies
    demasked_aggregated_randomness = np.mod( aggregated_masked_randomness - aggregated_mask_for_randomness, p)
    demasked_aggregated_hashes = aggregated_masked_hashed + (-1* aggregated_mask_for_hashes)  # 

    # aggregate the commits
    compute_elgamal_commitment_to_hash_randomness(demasked_aggregated_hashes, demasked_aggregated_randomness)

    # aggregate the gradients
    aggregated_gradients = sum([gradients])

    assert True, "test"

    
test2()
    






#todo: input dimensionality for sample model for lightsecagg from euklid
      





