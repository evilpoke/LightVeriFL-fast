#from ....fedml_api

from lightverirepo.fedml_api.distributed.lightveriagg.mpc_function import model_masking
from lightverirepo.fedml_api.distributed.lightveriagg.utils import transform_tensor_to_finite
from lightverirepo.fedml_api.distributed.utils.EC import PI_addEC, add_field_elements, compute_elgamal_commitment_to_hash_randomness, generate_point_EC, hash_a_sample_gradient
import numpy as np


"""

    IGNORE THIS TESTING

"""


def test1():
    p = 2147483647
    d = int(600372.0)
    local_mask = np.random.randint(p, size=(d, 1))
    print(local_mask)


def different_generators():
    v = np.array([2,6], dtype=int)  # TODO: nothing up my sleeve
    distinct_bases = distinct_point_compute(v)
    return distinct_bases





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
      





