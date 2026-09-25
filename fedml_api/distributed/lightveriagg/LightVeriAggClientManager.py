import logging
import os
import pickle
import sys

from fastecdsa.curve import P256
from fedml_api.distributed.utils.EC import LightVeriFL_enc_EC, PI_addEC, generate_Pedersen_commitment, generate_hash, generate_point_EC
import numpy as np
import time
from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.client.client_manager import ClientManager
from .message_define import MyMessage
from .utils import transform_list_to_tensor, transform_tensor_to_finite
from .utils import model_dimension
from .mpc_function import model_masking, mask_encoding, compute_aggregate_encoded_mask
from fastecdsa.point import Point

class LightVeriAggClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.local_mask = None
        self.encoded_mask_dict = dict()
        self.flag_encoded_mask_dict = dict()
        self.worker_num = size - 1
        self.timing_measurements = dict()
        self.dimensions = []
        self.total_dimension = None                     # is initialized with the model init
        for idx in range(self.worker_num):
            self.flag_encoded_mask_dict[idx] = False


        # new added parameters in main file
        self.targeted_number_active_clients = args.targeted_number_active_clients
        self.privacy_guarantee = args.privacy_guarantee
        self.prime_number = args.prime_number
        self.precision_parameter = args.precision_parameter
        self.encoded_veri_mask_dict = {}

        # verification
        self.alpha = np.zeros((1,), dtype=int)  # TODO: this can only be wrong
        self.agg_client_hashes_epochs = [0]*self.num_rounds
        self.N = self.size - 1
        self.stored_model = None
        self.hz = None
        self.rz = None
        self.all_commmitments = [0]*self.N
        self.encoded_veri_mask_dict = dict()
        self.flag_encoded_veri_mask_dict = dict()
        self.local_sample_num = -1
        self.check_second_veri_every_amount = self.num_rounds
        self.locally_accumulated_model = None
        self.agg_client_hashes_epochs_complete = self.num_rounds*[0]


    def run(self):
        super().run()

    def register_message_receive_handlers(self):


        #    - client generates zv_i , generates [nv_i]_js , encode to [\tilde zv_i]_j
        #   -> 2 client sends [\tilde zv_i]_j to other clients    
        
        

        # - receive model params from server
        # - feed the model parameters into the local model
        # - generate mask
        # - encode local mask to \tilde z_i j
        # - send corresponding mask DIRECTLY to ONLY the respective other clients
        #
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init)

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_H_R_TO_CLIENT, self.handle_message_hash_randomess_aggreg
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_VERI_COMMITMENT_TO_CLIENT_FWD, self.handle_message_vericomfrwd
        )

        # quantize the model 
        # some internal logic checks
        # send x+z to server
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_BOTH_ENCODED_MASK_TO_CLIENT_FWD, self.handle_message_receive_encoded_mask_from_server
        )


        # We receive from the server the current global model
        # We then load in the current global model
        # We trade with all other clients directly \tilde z_ij  (self.__offline() ) 
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.handle_message_receive_model_from_server
        )


        # - receive some keys
        # - (clients send the aggregate mask to the server for resolving)
        #
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SEND_TO_ACTIVE_CLIENT, self.handle_message_receive_active_from_server
        )


    def handle_message_vericomfrwd(self, msg_params):
        logging.info("client %d handle_message_vericomfrwd from server." % self.get_sender_id())
        commitment_from_client =      msg_params.get(MyMessage.MSG_ARG_KEY_COMMITMENT)
        client_id = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_ID)
        self.add_commitment(client_id - 1, commitment_from_client)
        b_all_received = self.flag_commitment_dict()
        if b_all_received:
            logging.info("client %d has received all commitments" % self.get_sender_id())

            # now uploading masked hash and masked randomness
            t0 = time.time()
            self.send_veri_masked_hash_randomness()
            t1 = time.time()
            self.timing_measurements["handle_message_vericomfrwd: sending hz and rz"] = t1-t0

            # Mask the local model
            t0 = time.time()
            weights_finite = self.stored_model
            masked_weights = model_masking(weights_finite, self.dimensions, self.local_mask, self.prime_number)
            t1 = time.time()
            self.timing_measurements["handle_message_vericomfrwd: masking model with x_i + z_i"] = t1-t0

            if self.local_sample_num == -1:
                raise Exception("No training has happened yet.")

            t0 = time.time()
            self.send_model_to_server(0, masked_weights, self.local_sample_num)
            t1 = time.time()
            self.timing_measurements["handle_message_vericomfrwd: sending masked model to server"] = t1-t0
            

    def handle_message_init(self, msg_params):
        # gets initial global model from the federator
        logging.info("client %d handle_message_init from server." % self.get_sender_id())

        t0 = time.time()
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        if self.args.is_mobile == 1:
            global_model_params = transform_list_to_tensor(global_model_params)

        # inits new model by these new global params
        self.dimensions, self.total_dimension = model_dimension(global_model_params)
        self.alpha = np.zeros((self.total_dimension,), dtype=int)
        t1 = time.time()
        self.timing_measurements["handle_message_init: Quantisation"] = t1-t0

        # updates the model
        t0 = time.time()
        self.trainer.update_model(global_model_params)
        # TODO ???
        self.trainer.update_dataset(int(client_index))
        t1 = time.time()
        self.timing_measurements["handle_message_init: Updating model and dataset"] = t1-t0

        # TODO :??
        self.round_idx = 0
        self.__offline()


    def handle_message_receive_encoded_mask_from_server(self, msg_params):
        
        encoded_mask = msg_params.get(MyMessage.MSG_ARG_KEY_ENCODED_MASK)
        encoded_veri_mask = msg_params.get(MyMessage.MSG_ARG_KEY_VERI_ENCODED_MASK)

        client_id = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_ID)
        logging.info(
            "Client %d receive encoded_mask = %s from Client %d" % (self.get_sender_id(), encoded_mask, client_id)
        )
        self.add_encoded_mask(client_id - 1, encoded_mask)
        self.add_encoded_veri_mask(client_id-1, encoded_veri_mask)
        b_all_received = self.check_whether_all_encoded_mask_receive()
        if b_all_received:
            t0 = time.time()
            # train, QUANTISATION , masking , sending
            x = self.__train()
            t1 = time.time()
            self.timing_measurements["handle_message_receive_encoded_mask_from_server: actual training"] = t1-t0

            # store x
            self.stored_model = x
            t0 = time.time()
            # generate hash based on x (and randomness)
            h = generate_hash(x,  self.alpha, self.total_dimension)
            t1 = time.time()
            self.timing_measurements["handle_message_receive_encoded_mask_from_server: generating h_i"] = t1-t0

            t0 = time.time()
            Pedersen_noise = generate_point_EC()

            # masking the hash and randomness
            self.hz = h + self.local_veri_mask
            self.rz = Pedersen_noise + self.local_veri_mask
            t1 = time.time()
            self.timing_measurements["handle_message_receive_encoded_mask_from_server: masking h_i and r_i"] = t1-t0

            # create commitment
            t0 = time.time()

            Pedersen_coeff = np.zeros((2,), dtype=int)
            curve_g = Point(P256.gx, P256.gy, curve=P256)

            Pedersen_g = (Pedersen_coeff[0] * curve_g)
            Pedersen_l = (Pedersen_coeff[1] * curve_g)
            commitment = generate_Pedersen_commitment(h, Pedersen_g, Pedersen_l, Pedersen_noise)            
            self.all_commmitments[self.get_sender_id] = h
            t1 = time.time()
            self.timing_measurements["handle_message_receive_encoded_mask_from_server: generating commit and storing"] = t1-t0


            # send commitment to server for broadcast
            t0 = time.time()
            self.send_veri_commitment(commitment)
            t1 = time.time()
            self.timing_measurements["handle_message_receive_encoded_mask_from_server: sending commit to other clients"] = t1-t0

            """        NOPE below!
            for receive_id in range(1, self.size):
                encoded_mask = encoded_mask_set[receive_id - 1]
                encoded_veri_mask = encoded_veri_mask_set[receive_id - 1]
                if receive_id != self.get_sender_id():

                    self.send_veri_encoded_mask_to_server(receive_id, encoded_veri_mask) # MSG_TYPE_C2S_SEND_ENCODED_MASK_TO_SERVER
                    self.send_encoded_mask_to_server(receive_id, encoded_mask)

                else:
                    self.encoded_mask_dict[receive_id - 1] = encoded_mask
                    self.flag_encoded_mask_dict[receive_id - 1] = True


            :"""

    

    def handle_message_hash_randomess_aggreg(self, msg_params):
        # ----------------
        # - Get hash and random agg from server.
        # - checking the integrity of aggregation
        # - Starting the new round, beginning with    
        #    [Separate verification mask]
        #       - client generates zv_i , generates [nv_i]_js , encode to [\tilde zv_i]_j
        #       -> 2 (client sends [\tilde zv_i]_j to other clients via server)            MSG_TYPE_C2S_SEND_VERI_ENCODED_MASK_TO_SERVER 
        # ----------------
        t0 = time.time()
        N = self.size - 1
        logging.info("client %d handle_message_hash_randomess_aggreg." % self.get_sender_id())
        aggregate_hash_U = msg_params.get(MyMessage.MSG_ARG_KEY_AGGREGATE_HASH_FOR_U)
        aggregate_hash_D = msg_params.get(MyMessage.MSG_ARG_KEY_AGGREGATE_HASH_FOR_D)
        aggregate_rnd_U = msg_params.get(MyMessage.MSG_ARG_KEY_AGGREGATE_RND_FOR_U)
        aggregate_rnd_D = msg_params.get(MyMessage.MSG_ARG_KEY_AGGREGATE_RND_FOR_D)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        
        self.current_i_trial += 1
        self.agg_client_hashes_epochs_U[self.current_i_trial] = aggregate_hash_U

        self.agg_client_hashes_epochs_complete[self.current_i_trial] = aggregate_hash_U + aggregate_hash_D
        t1 = time.time()
        self.timing_measurements["handle_message_hash_randomess_aggreg: receiving and adding and storing hashes"] = t1-t0

        t0 = time.time()
        Pedersen_coeff = np.zeros((2,), dtype=int)
        curve_g = Point(P256.gx, P256.gy, curve=P256)
        
        Pedersen_g = (Pedersen_coeff[0] * curve_g)
        Pedersen_l = (Pedersen_coeff[1] * curve_g)
        self_commitments_agg_U = generate_Pedersen_commitment(aggregate_hash_U, Pedersen_g, Pedersen_l, aggregate_rnd_U)
        self_commitments_agg_D = generate_Pedersen_commitment(aggregate_hash_D, Pedersen_g, Pedersen_l, aggregate_rnd_D)

        # actually summing up the commitments
        #self.local_sample_num
        
        surviving_commitments = []
        non_surviving_commitments = []
        for i in range(self.size):
            if i in self.active_clients_first_round:
                surviving_commitments.append(self.all_commmitments[i])
            else:
                non_surviving_commitments.append(self.all_commmitments[i])

        commitments_summed = PI_addEC(surviving_commitments)
        commitments_summed = (commitments_summed + (-(N - 1) * (Pedersen_g + Pedersen_l)))
        if self_commitments_agg_U == commitments_summed:
            logging.info("client %d confirmed correct commit aggregation" % self.get_sender_id())
        else:
            logging.info("error")
            exit(1)
            
        commitments_summed = PI_addEC(non_surviving_commitments)
        commitments_summed = (commitments_summed + (-(N - 1) * (Pedersen_g + Pedersen_l)))
        if self_commitments_agg_D == commitments_summed:
            logging.info("client %d confirmed correct commit aggregation" % self.get_sender_id())
        else:
            logging.info("error")
            exit(1)
        t1 = time.time()
        self.timing_measurements["handle_message_hash_randomess_aggreg: commitments verified"] = t1-t0
        
        if self.round_idx % self.check_second_veri_every_amount == 0:
            t0 = time.time()
            h_agg = generate_hash(self.locally_accumulated_model)
            #h_agg = generate_hash([sum(x) for x in zip(*server_agg_gradient_epochs)], alpha, d)
            
            
            h_blindly_added = PI_addEC(self.agg_client_hashes_epochs_complete)

            if h_agg == h_blindly_added:
                logging.info("client %d confirmed correct hash aggregation" % self.get_sender_id())
            else:
                logging.info("error")
                exit(1)
            t1 = time.time()
            self.timing_measurements["handle_message_hash_randomess_aggreg: aggregate model verified"] = t1-t0

        # Collecting timing data
        time_out = []
        time_out.append(self.timing_measurements)
        path = "./results/LightVeriSecFL_client_" + str(self.get_sender_id()) + "_round_" + str(self.round_idx) + "_L_"+str(self.check_second_veri_every_amount)+"_N_"+str(self.size) 
        pickle.dump(time_out, open(path, 'wb'), -1)
        self.timing_measurements = dict()
        
        # Restarting to the next round. Note that the federator might have already terminated the training 
        if self.round_idx == self.num_rounds:
            time.sleep(5)
            exit(0)
        logging.info("client %d starts a new FL round..." % self.get_sender_id())
        self.__offline()

        
        

    def handle_message_receive_model_from_server(self, msg_params):
        # ----------------
        # Get the model update from the server
        # ----------------
        
        t0 = time.time()
        logging.info("client %d handle_message_receive_model_from_server." % self.get_sender_id())
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        t1 = time.time()
        self.timing_measurements["handle_message_receive_model_from_server: receiving model"] = t1-t0

        t0 = time.time()
        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(model_params)

        # training update the local model
        self.trainer.update_model(model_params)
        
        # TODO ??
        self.trainer.update_dataset(int(client_index))
        self.round_idx += 1

        # masking and then sending just the mask. NOT THE ACTUAL MASKED MODEL!!
        #self.__offline() NOT HERE!!!

        if self.round_idx == self.num_rounds - 1:
            logging.info("this client has finished the training!")

        t1 = time.time()
        self.timing_measurements["handle_message_receive_model_from_server: mobile, updating model, dataset"] = t1-t0

        # sending the veri mask for resolving
        
        t0 = time.time()
        aggregate_veri_mask = compute_aggregate_encoded_mask(self.encoded_veri_mask_dict, self.prime_number, self.active_clients_first_round)
        t1 = time.time()
        self.timing_measurements["handle_message_receive_model_from_server: generating aggregate sum: [z_i]j"] = t1-t0

        t0 = time.time()
        self.send_veri_aggregate_mask(0, aggregate_veri_mask)
        t1 = time.time()
        self.timing_measurements["handle_message_receive_model_from_server: sending aggregate sum: [z_i]j"] = t1-t0


    def handle_message_receive_active_from_server(self, msg_params):
        # sending the mask
        
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        # Receive the set of active client id in first round
        active_clients_first_round = msg_params.get(MyMessage.MSG_ARG_KEY_ACTIVE_CLIENTS)
        self.active_clients_first_round = active_clients_first_round
        logging.info(
            "Client %d receive active_clients in the first round = %s"
            % (self.get_sender_id(), active_clients_first_round)
        )
        t0 = time.time()
        # Compute the aggregate of encoded masks for the active clients
        p = self.prime_number
        aggregate_encoded_mask = compute_aggregate_encoded_mask(self.encoded_mask_dict, p, active_clients_first_round)
        t1 = time.time()
        self.timing_measurements["handle_message_receive_active_from_server: compute the aggregate sum z_ij for active clients"] = t1-t0

        # Send the aggregate of encoded mask to server
        t0 = time.time()
        self.send_aggregate_encoded_mask_to_server(0, aggregate_encoded_mask)
        t1 = time.time()
        self.timing_measurements["handle_message_receive_active_from_server: sending the aggregate sum z_ij"] = t1-t0

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def send_veri_commitment(self, commitment):

        message = Message(MyMessage.MSG_TYPE_C2S_VERI_COMMITMENT_TO_SERVER, self.get_sender_id(), 0)
        message.add_params(MyMessage.MSG_ARG_KEY_COMMITMENT, commitment)
        self.send_message(message)
   

    def send_veri_encoded_mask_to_server(self, receive_id, encoded_veri_mask_set):
        # send this (also) to the respective client

        message = Message(MyMessage.MSG_TYPE_C2S_SEND_VERI_ENCODED_MASK_TO_SERVER, self.get_sender_id(), 0)
        message.add_params(MyMessage.MSG_ARG_KEY_VERI_ENCODED_MASK, encoded_veri_mask_set)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_ID, receive_id)
        self.send_message(message)

    def send_veri_masked_hash_randomness(self):

        message = Message(MyMessage.MSG_TYPE_C2S_VERI_MASKED_HASH_R, self.get_sender_id(), 0)
        message.add_params(MyMessage.MSG_ARG_KEY_MASKED_HASH, self.hz)
        message.add_params(MyMessage.MSG_ARG_KEY_MASKED_RND,  self.rz)
        self.send_message(message)

           

    def send_encoded_mask_to_server(self, receive_id, encoded_mask):
        # send this only to the respective client
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_ENCODED_MASK_TO_SERVER, self.get_sender_id(), 0)
        message.add_params(MyMessage.MSG_ARG_KEY_ENCODED_MASK, encoded_mask)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_ID, receive_id)
        self.send_message(message)

    def send_veri_aggregate_mask(self, receive_id, aggregate_veri_mask_U, aggregate_veri_mask_D):
        message = Message(MyMessage.MSG_TYPE_C2S_VERI_SEND_MASK_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_AGGREGATE_VERI_MASK_SURVIVING, aggregate_veri_mask_U)
        message.add_params(MyMessage.MSG_ARG_KEY_AGGREGATE_VERI_MASK_DROPPING, aggregate_veri_mask_D)

        self.send_message(message)

    def send_aggregate_encoded_mask_to_server(self, receive_id, aggregate_encoded_mask):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MASK_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_AGGREGATE_ENCODED_MASK, aggregate_encoded_mask)
        self.send_message(message)

    def add_encoded_veri_mask(self, index, encoded_veri_mask):
        self.encoded_veri_mask_dict[index] = encoded_veri_mask
        self.flag_encoded_veri_mask_dict[index] = True

    def add_commitment(self, index, commitment):

        self.all_commmitments[index] = commitment

    def add_encoded_mask(self, index, encoded_mask):
        self.encoded_mask_dict[index] = encoded_mask
        self.flag_encoded_mask_dict[index] = True

    def check_whether_all_commitments_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_commitment_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_commitment_dict[idx] = False
        return True

    def check_whether_all_encoded_mask_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_encoded_mask_dict[idx]:
                return False
            if not self.flag_encoded_veri_mask_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_encoded_mask_dict[idx] = False
            self.flag_encoded_veri_mask_dict[idx] = False
        return True

    def encoded_mask_sharing(self, encoded_mask_set, encoded_veri_mask_set):
        for receive_id in range(1, self.size):
            encoded_mask = encoded_mask_set[receive_id - 1]
            encoded_veri_mask = encoded_veri_mask_set[receive_id - 1]
            if receive_id != self.get_sender_id():

                self.send_veri_encoded_mask_to_server(receive_id, encoded_veri_mask) # MSG_TYPE_C2S_SEND_ENCODED_MASK_TO_SERVER
                self.send_encoded_mask_to_server(receive_id, encoded_mask)

            else:
                self.encoded_mask_dict[receive_id - 1] = encoded_mask
                self.flag_encoded_mask_dict[receive_id - 1] = True

    def __offline(self):
        # - Creating a local mask z and zv
        # - Sharding z and zv
        # - sending z_ij and zv_ij to the respective client
        
        logging.info("#######Client %d offline encoding round_id = %d######" % (self.get_sender_id(), self.round_idx))
        t0 = time.time()
        
        # encoded_mask_set = self.mask_encoding()
        d = self.total_dimension
        N = self.size - 1
        U = self.targeted_number_active_clients
        T = self.privacy_guarantee
        p = self.prime_number
        logging.debug("d = {}, N = {}, U = {}, T = {}, p = {}".format(d, N, U, T, p))

        n_i = np.random.randint(p, size=(T * d // (U - T), 1))
        logging.debug("size of n_i is: " + str((T * d // (U - T), 1)))

        logging.debug("expected full shape is " + str((U, d // (U - T))))
        
        self.local_mask = np.random.randint(p, size=(d, 1))                     # mask z_i for client i
        self.local_veri_mask = generate_point_EC()
        self.n_array = []  # n_array comes from the EC
        for x in range(T):
            self.n_array.append(generate_point_EC())
        
        # mask_encoding will: 
        #   split this into z_ij    
        #   randomly sample n_ij..
        #   multiplies this with W ?!
        #   
        encoded_mask_set = mask_encoding(d, N, U, T, p, self.local_mask)
        
        alpha_s = np.array(range(N)) + 1
        beta_s = np.array(range(U)) + (N+1)
        #  zv_tilde_array =
        encoded_veri_mask_set = LightVeriFL_enc_EC( self.local_veri_mask, self.n_array, alpha_s, beta_s, P256)   #mask_encoding(1, N, U, T, p, self.local_veri_mask)      # TODO: replace with EC encoding: done?

        t1 = time.time()
        self.timing_measurements["__offline: z and zv generated, sharded"] = t1-t0
                   
        # Send the encoded masks to other clients (via server)
        t0 = time.time()
        self.encoded_mask_sharing(encoded_mask_set, encoded_veri_mask_set)
        t1 = time.time()
        self.timing_measurements["__offline: sharded z, zv distributed"] = t1-t0
        # TODO: encoded_veri_mask_set sharing?? Done?!

    def __train(self):
        logging.info("Client %d #######training########### round_id = %d" % (self.get_sender_id(), self.round_idx))
        weights, local_sample_num = self.trainer.train(self.round_idx)
        self.local_sample_num = local_sample_num
        # Convert the model from real to finite
        p = self.prime_number
        q_bits = self.precision_parameter
        weights_finite = transform_tensor_to_finite(weights, p, q_bits)

        return weights_finite