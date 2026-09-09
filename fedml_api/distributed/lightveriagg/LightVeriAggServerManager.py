import logging
import pickle
from time import sleep

from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.server.server_manager import ServerManager
from .message_define import MyMessage
from .utils import transform_tensor_to_list
import time

class LightVeriAggServerManager(ServerManager):
    def __init__(
        self,
        args,
        aggregator,
        comm=None,
        rank=0,
        size=0,
        backend="MPI",
        is_preprocessed=False,
        preprocessed_client_lists=None,
    ):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists
        self.timing_measurements = dict()

        self.active_clients_first_round = []
        self.active_clients_second_round = []

        ### new added parameters in main file ###
        self.targeted_number_active_clients = args.targeted_number_active_clients
        self.privacy_guarantee = args.privacy_guarantee
        self.prime_number = args.prime_number
        self.precision_parameter = args.precision_parameter

        ### verification
        self.masked_hash_rnd_dict = dict()
        self.encoded_veri_mask_dict = dict()
        self.client_num_in_total = self.size
        for i in range(self.client_num_in_total):
            self.encoded_veri_mask_dict[i] = dict()
            

    def run(self):
        super().run()

    def send_init_msg(self):
        # sampling clients
        client_indexes = self.aggregator.client_sampling(
            self.round_idx, self.args.client_num_in_total, self.args.client_num_per_round
        )
        global_model_params = self.aggregator.get_global_model_params()
        self.aggregator.get_model_dimension(global_model_params)

        t0 = time.time()
        if self.args.is_mobile == 1:
            global_model_params = transform_tensor_to_list(global_model_params)
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, global_model_params, client_indexes[process_id - 1])
        t1 = time.time()
        self.timing_measurements["send_init_msg: sending initial model"] = t1-t0


    def register_message_receive_handlers(self):

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_VERI_MASKED_HASH_R, self.handle_message_receive_veri_masked_hash_r
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_VERI_SEND_MASK_TO_SERVER, self.handle_message_receive_veri_aggregate_mask
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_VERI_COMMITMENT_TO_SERVER, self.handle_message_forwarding_commitments
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_VERI_ENCODED_MASK_TO_SERVER, self.handle_message_receive_veri_encoded_mask_from_client
        )

        
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_ENCODED_MASK_TO_SERVER, self.handle_message_receive_encoded_mask_from_client
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.handle_message_receive_model_from_client
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MASK_TO_SERVER, self.handle_message_receive_aggregate_encoded_mask_from_client
        )
        print("final registration")

    def handle_message_forwarding_commitments(self, msg_params):
        # TODO?: waiting for all commits to be collected
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        receive_id = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_ID)
        commitment = msg_params.get(MyMessage.MSG_ARG_KEY_COMMITMENT)
        
        self.send_commitment_to_client(sender_id, receive_id, commitment)
        
    def handle_message_receive_veri_masked_hash_r(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        masked_hash_U = msg_params.get(MyMessage.MSG_ARG_KEY_MASKED_HASH_U)
        masked_hash_D = msg_params.get(MyMessage.MSG_ARG_KEY_MASKED_HASH_D)

        masked_rnd_U = msg_params.get(MyMessage.MSG_ARG_KEY_MASKED_RND_U)
        masked_rnd_D = msg_params.get(MyMessage.MSG_ARG_KEY_MASKED_RND_D)


        self.masked_hash_rnd_dict[sender_id] = (masked_hash_U, masked_hash_D, masked_rnd_U, masked_rnd_D)


    def handle_message_receive_veri_encoded_mask_from_client(self, msg_params):
        # receive the encoded [\tilde z_i]_j from i
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        receive_id = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_ID)
        encoded_mask = msg_params.get(MyMessage.MSG_ARG_KEY_VERI_ENCODED_MASK)
        self.encoded_veri_mask_dict[sender_id][receive_id] = encoded_mask
        #self.send_message_encoded_mask_to_client(sender_id, receive_id, encoded_mask) waiting until second encoding mask comes in


    def handle_message_receive_encoded_mask_from_client(self, msg_params):  # not sum
        # receive the encoded [\tilde z_i]_j from i
        # this happens after the veri encoded mask was send by the server

        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        receive_id = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_ID)
        encoded_mask = msg_params.get(MyMessage.MSG_ARG_KEY_ENCODED_MASK)

        respective_encoded_veri_mask = self.encoded_veri_mask_dict[sender_id][receive_id]
        
        self.send_both_message_encoded_mask_to_client(sender_id, receive_id, respective_encoded_veri_mask, encoded_mask)

    def handle_message_receive_model_from_client(self, msg_params):
        # Receive the masked models from clients
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_TYPE_C2S_AGG_MASKED_X)

        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number)
        self.active_clients_first_round.append(sender_id - 1)
        b_all_received = self.aggregator.check_whether_all_receive()
        if b_all_received:
            logging.info("Server: all models received in round_idx %d" % self.round_idx)
            # Specify the active clients for the first round and inform them
            t0 = time.time()
            for receiver_id in range(1, self.size):
                self.send_message_to_active_client(receiver_id, self.active_clients_first_round)
            t1 = time.time()
            self.timing_measurements["handle_message_receive_model_from_client: notifying U_1"] = t1-t0


    def handle_message_receive_veri_aggregate_mask(self, msg_params):
        # Receive the aggregate of encoded masks for active clients
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        aggregate_veri_encoded_mask_U = msg_params.get(MyMessage.MSG_ARG_KEY_AGGREGATE_VERI_MASK_SURVIVING)
        aggregate_veri_encoded_mask_D = msg_params.get(MyMessage.MSG_ARG_KEY_AGGREGATE_VERI_MASK_DROPPED)
        self.aggregator.add_local_veri_aggregate_encoded_mask_U(sender_id - 1, aggregate_veri_encoded_mask_U)
        self.aggregator.add_local_veri_aggregate_encoded_mask_D(sender_id - 1, aggregate_veri_encoded_mask_D)

        logging.info("Server handle_message_receive_veri_aggregate_mask")

        # Active clients for the second round
        self.active_clients_veri_second_round.append(sender_id - 1)
        b_all_received = self.aggregator.check_whether_all_aggregate_veri_mask_receive()
        logging.info("Server: mask_all_received = " + str(b_all_received) + " in round_idx %d" % self.round_idx)

        # After receiving enough aggregate of encoded masks, server recovers the aggregate-model
        if b_all_received:
            # Secure Model Aggregation
            t0 = time.time()
            aggregate_veri_mask_U = self.aggregator.aggregate_veriaggmask_reconstruction_U(
                self.active_clients_first_round, self.active_clients_veri_second_round
            )
            t1 = time.time()
            self.timing_measurements["handle_message_receive_veri_aggregate_mask: aggregate zv for surviving"] = t1-t0

            t0 = time.time()
            aggregate_veri_mask_D = self.aggregator.aggregate_veriaggmask_reconstruction_D(
                self.active_clients_first_round, self.active_clients_veri_second_round
            )
            t1 = time.time()
            self.timing_measurements["handle_message_receive_veri_aggregate_mask: aggregate zv for dropped"] = t1-t0

            
            # start the next round
            self.round_idx += 1
            self.active_clients_first_round = []
            self.active_clients_second_round = []

            """ TODO  ??
            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    # sampling has already been done in data preprocessor
                    client_indexes = [self.round_idx] * self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.round_idx]
            else:
                # sampling clients
                client_indexes = self.aggregator.client_sampling(
                    self.round_idx, self.args.client_num_in_total, self.args.client_num_per_round
                )
            """
            
            client_indexes = self.client_indexes
            print("indexes of clients: " + str(client_indexes))
            print("size = %d" % self.size)

            """            if self.args.is_mobile == 1:
                global_model_params = transform_tensor_to_list(global_model_params)
            """

            t0 = time.time()
            (masked_hash_U, masked_hash_D, masked_rnd_U, masked_rnd_D) = self.masked_hash_rnd_dict[sender_id]
            
            aggregate_hash_U = masked_hash_U - aggregate_veri_mask_U
            aggregate_hash_D = masked_hash_D - aggregate_veri_mask_D
            aggregate_randomness_U = masked_rnd_U - aggregate_veri_mask_U
            aggregate_randomness_D = masked_rnd_D - aggregate_veri_mask_D
            t1 = time.time()
            self.timing_measurements["handle_message_receive_veri_aggregate_mask: demasking sum: h_i and sum: r_i"] = t1-t0

            t0 = time.time()
            for receiver_id in range(1, self.size):
                self.send_message_sync_veriagg_to_client(
                    receiver_id, aggregate_hash_U, aggregate_hash_D, aggregate_randomness_U, aggregate_randomness_D, client_indexes[receiver_id - 1]
                )
            t1 = time.time()
            self.timing_measurements["handle_message_receive_veri_aggregate_mask: sending aggregate hash and randomness to clients"] = t1-t0
            
            time_out = []
            time_out.append(self.timing_measurements)
            path = "./results/LightVeriSecFL_federator_round_" + str(self.round_idx) + "_L_"+str(self.check_second_veri_every_amount)+"_N_"+str(self.size) 
            pickle.dump(time_out, open(path, 'wb'), -1)
            self.timing_measurements = dict()

            # New round starts if this following if-clause does not trigger and allows the clients to again trigger a new handle
            if self.round_idx == self.round_num:
                logging.info("==============================\n SERVER: TRAINING IS FINISHED! \n The final model update has occurred and the final verification aggregate mask was sent.")
                sleep(3)
                self.finish()


    def handle_message_receive_aggregate_encoded_mask_from_client(self, msg_params):  # sum
        # Receive the aggregate of encoded masks for active clients
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        aggregate_encoded_mask = msg_params.get(MyMessage.MSG_ARG_KEY_AGGREGATE_ENCODED_MASK)
        self.aggregator.add_local_aggregate_encoded_mask(sender_id - 1, aggregate_encoded_mask)
        logging.info(
            "Server handle_message_receive_aggregate_mask = %d from_client =  %d"
            % (len(aggregate_encoded_mask), sender_id)
        )
        # Active clients for the second round
        self.active_clients_second_round.append(sender_id - 1)
        b_all_received = self.aggregator.check_whether_all_aggregate_encoded_mask_receive()
        logging.info("Server: mask_all_received = " + str(b_all_received) + " in round_idx %d" % self.round_idx)

        # After receiving enough aggregate of encoded masks, server recovers the aggregate-model
        if b_all_received:
            # Secure Model Aggregation
            t0 = time.time()
            global_model_params = self.aggregator.aggregate_model_reconstruction(
                self.active_clients_first_round, self.active_clients_second_round
            )
            t1 = time.time()
            self.timing_measurements["handle_message_receive_aggregate_encoded_mask_from_client: constructing global model from received sum: [z_i]j"] = t1-t0


            # evaluation
            t0 = time.time()
            self.aggregator.test_on_server_for_all_clients(self.round_idx)
            t1 = time.time()
            self.timing_measurements["handle_message_receive_aggregate_encoded_mask_from_client: eval"] = t1-t0

            t0 = time.time()
            # start the next round
            self.round_idx += 1
            self.active_clients_first_round = []
            self.active_clients_second_round = []

            if self.round_idx == self.round_num:
                logging.info("Last model came in. Server will soon send out the last verification")

            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    # sampling has already been done in data preprocessor
                    client_indexes = [self.round_idx] * self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.round_idx]
            else:
                # sampling clients
                client_indexes = self.aggregator.client_sampling(
                    self.round_idx, self.args.client_num_in_total, self.args.client_num_per_round
                )
            self.client_indexes = client_indexes

            print("indexes of clients: " + str(client_indexes))
            print("size = %d" % self.size)
            if self.args.is_mobile == 1:
                global_model_params = transform_tensor_to_list(global_model_params)
            t1 = time.time()
            self.timing_measurements["handle_message_receive_aggregate_encoded_mask_from_client: artificial sampling"] = t1-t0


            t0 = time.time()
            for receiver_id in range(1, self.size):
                self.send_message_sync_model_to_client(
                    receiver_id, global_model_params, client_indexes[receiver_id - 1]
                )
            t1 = time.time()
            self.timing_measurements["handle_message_receive_aggregate_encoded_mask_from_client: forwarding global model to clients"] = t1-t0


    def send_message_init_config(self, receive_id, global_model_params, client_index):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        
        self.send_message(message)


    def send_both_message_encoded_mask_to_client(self, sender_id, receive_id,respective_encoded_veri_mask , encoded_mask):
        message = Message(MyMessage.MSG_TYPE_S2C_BOTH_ENCODED_MASK_TO_CLIENT_FWD, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_ID, sender_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ENCODED_MASK, encoded_mask)
        message.add_params(MyMessage.MSG_ARG_KEY_VERI_ENCODED_MASK, respective_encoded_veri_mask)
        
        self.send_message(message)

    def send_commitment_to_client(self, sender_id, receive_id, commitment):
        logging.info("Server send_commitment_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_VERI_COMMITMENT_TO_CLIENT_FWD, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_COMMITMENT, commitment)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_ID, sender_id)
        self.send_message(message)

    def send_message_sync_veriagg_to_client(self,
                    receive_id, 
                    aggregate_hash_U, 
                    aggregate_hash_D, 
                    aggregate_randomness_U, 
                    aggregate_randomness_D, 
                    client_index
                ):
        logging.info("Server send_message_sync_veriagg_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_H_R_TO_CLIENT, self.get_sender_id(), receive_id)

        message.add_params(MyMessage.MSG_ARG_KEY_AGGREGATE_HASH_FOR_U, aggregate_hash_U)
        message.add_params(MyMessage.MSG_ARG_KEY_AGGREGATE_HASH_FOR_D, aggregate_hash_D )
        message.add_params(MyMessage.MSG_ARG_KEY_AGGREGATE_RND_FOR_U,  aggregate_randomness_U)
        message.add_params(MyMessage.MSG_ARG_KEY_AGGREGATE_RND_FOR_D,  aggregate_randomness_D)

        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)        


    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index):
        logging.info("Server send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_to_active_client(self, receive_id, active_clients):
        logging.info("Server send_message_to_active_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SEND_TO_ACTIVE_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTIVE_CLIENTS, active_clients)
        self.send_message(message)
