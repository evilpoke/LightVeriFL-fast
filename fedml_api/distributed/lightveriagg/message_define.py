

# this will use LightSegAgg for aggregation and LightVeriFL for verification
# we use separate masks z_i r_i
# TODO: use just one mask

class MyMessage(object):
    r"""
    LightSecAgg Protocol (+ LightVeriFL verification with an El Gamal commitment):
   -> 1 (server initializes the model parameters)    MSG_TYPE_S2C_INIT_CONFIG

   
   [Separate verification mask]
   - client generates zv_i = (zh_i, zr_i):  zh_i a random EC point (masks the hash h_i),
                                           zr_i a random integer in Z_n (masks the El Gamal randomness r_i; n = group order)
   - generates T blindings [nv_i]_k (EC points resp. integers), encodes both with the (N, T+1) MDS code to [\tilde zv_i]_j = ([\tilde zh_i]_j, [\tilde zr_i]_j)
   -> 2 (client sends [\tilde zv_i]_j to other clients via server)            MSG_TYPE_C2S_SEND_VERI_ENCODED_MASK_TO_SERVER 
    
   [LightSeg mask]
   -  gen z_i  (in F_p^d, padded to a multiple of U-T)
   -  gen [n_i]_j
   -  encode to [\tilde z_i]_j with z_i and [n_i]_ks
   -> 3 (clients send [\tilde z_i]_j encoded mask to other clients)          MSG_TYPE_C2S_SEND_ENCODED_MASK_TO_SERVER

   [Forwarding verfi and LightSeg mask]
   -> 12 SERVER: forward [\tilde zv_i]_j and [\tilde z_i]_j to j             MSG_TYPE_S2C_BOTH_ENCODED_MASK_TO_CLIENT_FWD

   [Training]
   generate x  (quantized to F_p)
   
   [Veri Commit]
   - the client generates h_i based on x_i         (h_i = sum_k x_i[k] g_k, x_i[k] read as signed integer, g_k hash-to-curve)
   - the client randomly samples r_i in Z_n
   - the client generates commitment(h_i,r_i)      (El Gamal: c_i = (r_i G, h_i + r_i H))

   -> 4 (client sends the commitment c_i to other clients via server)      MSG_TYPE_C2S_VERI_COMMITMENT_TO_SERVER

   -> 13 SERVER: (forwards c_i to every other client)                      MSG_TYPE_S2C_VERI_COMMITMENT_TO_CLIENT_FWD
   
   [Veri Hash upload]  (once all commitments are exchanged)
   - generate \tilde h = h_i+zh_i (EC) and \tilde r = r_i+zr_i mod n
   -> 5 (client uploads \tilde r and \tilde h to server)                   MSG_TYPE_C2S_VERI_MASKED_HASH_R
                                                                            
   [LightSeg Finish]
   
   - generate \tilde x = x_i + z_i
   -> 6 (upload \tilde x to server)                                        MSG_TYPE_C2S_SEND_MODEL_TO_SERVER
 
   - SERVER: identifies surviving users U_1
   -> 7 (SERVER: sends set U_1)                                            MSG_TYPE_S2C_SEND_TO_ACTIVE_CLIENT

   - compute \sum_j\in U_1 [\tilde z_j]_i 
   -> 8 (send  \sum_j\in U_1 [\tilde z_j]_i to server)                     MSG_TYPE_C2S_SEND_MASK_TO_SERVER

        => adds clients that submit the agg mask to server as "active_clients_second_round"

        # TODO: introduce active_clients_third_round

   - SERVER: Recover \sum z_i (from U of the messages) and then \sum x_i
   -> 9 (SERVER: Broadcast \sum x_i = y in F_p; new global model = y / |U_1|)   MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT 
    
   - U = U_1 (from lightsegacc),  D = [N] \ U_1

   - each client i \in U (that did not drop) generates \sum_N [\tilde zv_j]_i, split into
     \sum_U [\tilde zv_j]_i (SURVIVING) and \sum_D [\tilde zv_j]_i (DROPPING)
   -> 10 (uploads \sum_N [\tilde zv_j]_i)                                  MSG_TYPE_C2S_VERI_SEND_MASK_TO_SERVER

   - SERVER: recovers \sum zv_i (one-shot, from any T+1 of the messages), \sum h_i , \sum r_i   (for U and D)
   -> 11 SERVER: broadcasts \sum h_i and \sum r_i to users U               MSG_TYPE_S2C_SYNC_H_R_TO_CLIENT

   - each user in U checks: 
      - COM(\sum h_i , \sum r_i ) =  \sum c_i          (for U and for D)

      
      - \sum_i\in U_1: (h_i ^l)  =: "h^l"
      - model in lth iteration: "y(l)"
      - private random coefficient "\alpha_l" (veri_alpha_bits bits; 0 -> \alpha_l = 1)
      
      in Lth iteration (every veri_batch_size rounds and in the last round)
      - \sum_l \in L:  \alpha_l h^l == hash( \sum \alpha_l y(l) )


    """

    # server to client                          
    MSG_TYPE_S2C_INIT_CONFIG = 1                  # done done
    MSG_TYPE_S2C_BOTH_ENCODED_MASK_TO_CLIENT_FWD = 12 # previously : MSG_TYPE_S2C_ENCODED_MASK_TO_CLIENT = 2 # client: done  # server: done
    MSG_TYPE_S2C_VERI_COMMITMENT_TO_CLIENT_FWD = 13    # client: done  server: done
    MSG_TYPE_S2C_SEND_TO_ACTIVE_CLIENT = 7     # client: done   server: done
    MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT = 9      # client: done   server: done
    MSG_TYPE_S2C_SYNC_H_R_TO_CLIENT = 11       # client: done   server: done
    
    # client to server
    MSG_TYPE_C2S_SEND_ENCODED_MASK_TO_SERVER = 2       # client: done
    MSG_TYPE_C2S_SEND_VERI_ENCODED_MASK_TO_SERVER = 3   # client: done
    MSG_TYPE_C2S_VERI_COMMITMENT_TO_SERVER = 4    # client: done   server: done
    MSG_TYPE_C2S_VERI_MASKED_HASH_R = 5           # client: done   server: done
    MSG_TYPE_C2S_SEND_MODEL_TO_SERVER = 6         # client: done   server: done
    MSG_TYPE_C2S_SEND_MASK_TO_SERVER = 8          # client: done   server: done
    MSG_TYPE_C2S_VERI_SEND_MASK_TO_SERVER = 10    # client: done   server: done
                  

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"


    """
        message payload keywords definition
    """
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_AGGREGATE_HASH_FOR_U = "aggregate_hash_sum_for_surviving"
    MSG_ARG_KEY_AGGREGATE_HASH_FOR_D = "aggregate_hash_sum_for_dropped"
    MSG_ARG_KEY_AGGREGATE_RND_FOR_U = "aggregate_rnd_sum_for_surviving"
    MSG_ARG_KEY_AGGREGATE_RND_FOR_D = "aggregate_rnd_sum_for_dropped"

    MSG_ARG_KEY_CLIENT_INDEX = "client_idx"

    MSG_ARG_KEY_TRAIN_CORRECT = "train_correct"
    MSG_ARG_KEY_TRAIN_ERROR = "train_error"
    MSG_ARG_KEY_TRAIN_NUM = "train_num_sample"

    MSG_ARG_KEY_TEST_CORRECT = "test_correct"
    MSG_ARG_KEY_TEST_ERROR = "test_error"
    MSG_ARG_KEY_TEST_NUM = "test_num_sample"

    MSG_ARG_KEY_ENCODED_MASK = "encoded_mask"
    MSG_ARG_KEY_VERI_ENCODED_MASK = "encoded_veri_mask"
    MSG_ARG_KEY_COMMITMENT = "veri_commitment"
    MSG_ARG_KEY_MASKED_HASH_U = "masked_hash_of_client"
    MSG_ARG_KEY_MASKED_HASH_D = "masked_hash_of_client"
    MSG_ARG_KEY_MASKED_RND_U = "masked_randomness_of_client"
    MSG_ARG_KEY_MASKED_RND_D = "masked_randomness_of_client"
    
    MSG_ARG_KEY_ACTIVE_CLIENTS = "active_clinets"
    MSG_ARG_KEY_AGGREGATE_ENCODED_MASK = "aggregate_encoded_mask"
    MSG_ARG_KEY_AGGREGATE_VERI_MASK_SURVIVING = "aggregate_veri_mask_surviving"
    MSG_ARG_KEY_AGGREGATE_VERI_MASK_DROPPING = "aggregate_veri_mask_dropped"

    MSG_ARG_KEY_CLIENT_ID = "client_id"


# this will use segagg for aggregation and LightVeriFL for verification
r"""
class MyMessage(object):
    #
    LightSecAgg Protocol:
       1 (server initializes the model parameters)
    
    -> 5 (clients send encoded mask to other clients via the server)
       gen z_i
       gen [n_i]_j
       encode to [\tilde z_i]_j with z_i and [n_i]_ks

       send [\tilde z_i]_j for all j to server

    -> 2 (the server transfers the encoded mask to clients)
    
       send to client j the mask [\tilde_i]_j from all clients i
    
    
    ==========the client is doing the model training and generates x_i =========
    
    - the client generates h_i based on x_i
    - the client randomly samples r_i
    - the client generates commitment(h_i,r_i)

    -> 8 (client sends the commitment c_i to server) [broadcast]

    -> 9 (server sends the commitment of all users to all users)

    - mask hash and randomness so to get \tilde h_i and \tilde r_i
    
    -> 10 (client send \tilde h_i and \tilde r_i to server)

    [one shot aggregate of y]

    -> 11 (server sends y to clients)
    
    - identify surviving users in verification phase: U
    
    -> 12 (servers sends client i a total of |U| messages of \sum_j\in N: [\tilde z_j]_i from user i...??)
      
    - clients compute \sum z_i  

    -> 13 ()

    

    -> 

    -> 6 (send the trained model to the server)
    -> 4 (the server asks the active users to upload the aggregate mask)
    -> 7 (clients send the aggregate mask to the server)

    ==========          model aggregation          =========

    -> 3 (the server send the aggregated model to all clients)

    

    # server to client
    MSG_TYPE_S2C_INIT_CONFIG = 1
    MSG_TYPE_S2C_ENCODED_MASK_TO_CLIENT = 2
    MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT = 3
    MSG_TYPE_S2C_SEND_TO_ACTIVE_CLIENT = 4

    # client to server
    MSG_TYPE_C2S_SEND_ENCODED_MASK_TO_SERVER = 5
    MSG_TYPE_C2S_SEND_MODEL_TO_SERVER = 6
    MSG_TYPE_C2S_SEND_MASK_TO_SERVER = 7

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    
        message payload keywords definition
    
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_CLIENT_INDEX = "client_idx"

    MSG_ARG_KEY_TRAIN_CORRECT = "train_correct"
    MSG_ARG_KEY_TRAIN_ERROR = "train_error"
    MSG_ARG_KEY_TRAIN_NUM = "train_num_sample"

    MSG_ARG_KEY_TEST_CORRECT = "test_correct"
    MSG_ARG_KEY_TEST_ERROR = "test_error"
    MSG_ARG_KEY_TEST_NUM = "test_num_sample"

    MSG_ARG_KEY_ENCODED_MASK = "encoded_mask"
    MSG_ARG_KEY_ACTIVE_CLIENTS = "active_clinets"
    MSG_ARG_KEY_AGGREGATE_ENCODED_MASK = "aggregate_encoded_mask"
    MSG_ARG_KEY_CLIENT_ID = "client_id"
"""