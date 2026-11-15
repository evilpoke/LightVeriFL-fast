import logging
import os
import pickle
from time import sleep

from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.server.server_manager import ServerManager
from fedml_api.distributed.utils.EC import PI_addEC, curve_n, ec_add
from fedml_api.distributed.utils.bench import timer
from .message_define import MyMessage
from .mpc_function import verification_dropouts
from .utils import save_model_dict_to_pickle, transform_tensor_to_list
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

        self.global_model_params = None
        self.modelpath = None
        self.set_path(getattr(args, "results_dir", "./results"))
        self.active_clients_first_round = []
        self.active_clients_second_round = []
        self.active_clients_veri_second_round = []
        self.client_indexes = None
        self.metrics = []                               # per round: test accuracy / loss

        ### new added parameters in main file ###
        self.targeted_number_active_clients = args.targeted_number_active_clients
        self.privacy_guarantee = args.privacy_guarantee
        self.prime_number = args.prime_number
        self.precision_parameter = args.precision_parameter
        self.veri_dropout_rate = float(getattr(args, "veri_dropout_rate", 0.0))
        self.veri_dropout_seed = int(getattr(args, "veri_dropout_seed", 0))

        ### verification
        self.masked_hash_rnd_dict = dict()
        self.encoded_veri_mask_dict = dict()
        self.client_num_in_total = self.size
        for i in range(self.client_num_in_total):
            self.encoded_veri_mask_dict[i] = dict()

    def set_path(self, basepath):
        self.modelpath = os.path.join(basepath, "lightveriagg_global_model.pkl")

    def run(self):
        super().run()

    def send_init_msg(self):
        # sampling clients
        client_indexes = self.aggregator.client_sampling(
            self.round_idx, self.args.client_num_in_total, self.args.client_num_per_round
        )
        self.client_indexes = client_indexes
        global_model_params = self.aggregator.get_global_model_params()
        self.aggregator.get_model_dimension(global_model_params)

        if self.args.is_mobile == 1:
            global_model_params = transform_tensor_to_list(global_model_params)
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, global_model_params, client_indexes[process_id - 1])


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

    def handle_message_forwarding_commitments(self, msg_params):
        # 4 -> 13: forward c_i to every other client
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        commitment = msg_params.get(MyMessage.MSG_ARG_KEY_COMMITMENT)
        for receive_id in range(1, self.size):
            if receive_id != sender_id:
                self.send_commitment_to_client(sender_id, receive_id, commitment)

    def handle_message_receive_veri_masked_hash_r(self, msg_params):
        # 5: \tilde h_i (EC point) and \tilde r_i (in Z_n)
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        masked_hash = msg_params.get(MyMessage.MSG_ARG_KEY_MASKED_HASH_U)   # (same key as ..._D)
        masked_rnd = msg_params.get(MyMessage.MSG_ARG_KEY_MASKED_RND_U)     # (same key as ..._D)
        self.masked_hash_rnd_dict[sender_id] = (masked_hash, masked_rnd)


    def handle_message_receive_veri_encoded_mask_from_client(self, msg_params):
        # 3: receive the encoded [\tilde zv_i]_j from i, kept until [\tilde z_i]_j arrives
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        receive_id = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_ID)
        encoded_mask = msg_params.get(MyMessage.MSG_ARG_KEY_VERI_ENCODED_MASK)
        self.encoded_veri_mask_dict[sender_id][receive_id] = encoded_mask


    def handle_message_receive_encoded_mask_from_client(self, msg_params):  # not sum
        # 2: receive the encoded [\tilde z_i]_j from i
        # this happens after the veri encoded mask was received (same sender -> FIFO)
        # -> 12: forward both to j

        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        receive_id = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_ID)
        encoded_mask = msg_params.get(MyMessage.MSG_ARG_KEY_ENCODED_MASK)

        respective_encoded_veri_mask = self.encoded_veri_mask_dict[sender_id].pop(receive_id)

        self.send_both_message_encoded_mask_to_client(sender_id, receive_id, respective_encoded_veri_mask, encoded_mask)

    def handle_message_receive_model_from_client(self, msg_params):
        # 6: Receive the masked models from clients
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number)
        self.active_clients_first_round.append(sender_id - 1)
        b_all_received = self.aggregator.check_whether_all_receive()
        if b_all_received:
            logging.info("Server: all models received in round_idx %d" % self.round_idx)
            # Specify the active clients for the first round (U_1) and inform them -> 7
            self.active_clients_first_round = sorted(self.active_clients_first_round)
            for receiver_id in range(1, self.size):
                self.send_message_to_active_client(receiver_id, self.active_clients_first_round)


    def handle_message_receive_veri_aggregate_mask(self, msg_params):
        # 10: Receive the aggregate of encoded verification masks of surviving (U = U_1) and dropped (D = [N]\U_1)
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        aggregate_veri_encoded_mask_U = msg_params.get(MyMessage.MSG_ARG_KEY_AGGREGATE_VERI_MASK_SURVIVING)
        aggregate_veri_encoded_mask_D = msg_params.get(MyMessage.MSG_ARG_KEY_AGGREGATE_VERI_MASK_DROPPING)
        self.aggregator.add_local_veri_aggregate_encoded_mask_U(sender_id - 1, aggregate_veri_encoded_mask_U)
        self.aggregator.add_local_veri_aggregate_encoded_mask_D(sender_id - 1, aggregate_veri_encoded_mask_D)

        # Active clients in the verification phase
        self.active_clients_veri_second_round.append(sender_id - 1)
        dropped = verification_dropouts(self.round_idx, self.active_clients_first_round, self.veri_dropout_rate,
                                        self.privacy_guarantee, self.veri_dropout_seed)
        expected = [c for c in self.active_clients_first_round if c not in dropped]
        b_all_received = len(self.active_clients_veri_second_round) >= len(expected)
        logging.info("Server: veri_mask_all_received = " + str(b_all_received) + " in round_idx %d" % self.round_idx)

        # After receiving the aggregated encoded verification masks, the server recovers sum h_i and sum r_i
        if b_all_received:
            responders = sorted(self.active_clients_veri_second_round)
            with timer("server: [V.0] decode aggregate verification masks, unmask sum h, sum r", who="server",
                       rnd=self.round_idx):
                # one-shot decoding from T+1 of the surviving users
                (aggregate_veri_mask_hash_U, aggregate_veri_mask_rnd_U) = \
                    self.aggregator.aggregate_veriaggmask_reconstruction_U(responders)
                (aggregate_veri_mask_hash_D, aggregate_veri_mask_rnd_D) = \
                    self.aggregator.aggregate_veriaggmask_reconstruction_D(responders)

                surviving = self.active_clients_first_round
                non_surviving = [i for i in range(self.size - 1)
                                 if i not in surviving and (i + 1) in self.masked_hash_rnd_dict]
                masked_hash_U = PI_addEC([self.masked_hash_rnd_dict[i + 1][0] for i in surviving])
                masked_rnd_U = sum(int(self.masked_hash_rnd_dict[i + 1][1]) for i in surviving) % curve_n
                masked_hash_D = PI_addEC([self.masked_hash_rnd_dict[i + 1][0] for i in non_surviving])
                masked_rnd_D = sum(int(self.masked_hash_rnd_dict[i + 1][1]) for i in non_surviving) % curve_n

                aggregate_hash_U = ec_add(masked_hash_U, -aggregate_veri_mask_hash_U)
                aggregate_hash_D = ec_add(masked_hash_D, -aggregate_veri_mask_hash_D)
                aggregate_randomness_U = (masked_rnd_U - aggregate_veri_mask_rnd_U) % curve_n
                aggregate_randomness_D = (masked_rnd_D - aggregate_veri_mask_rnd_D) % curve_n

            client_indexes = self.client_indexes
            # -> 11
            for receiver_id in range(1, self.size):
                self.send_message_sync_veriagg_to_client(
                    receiver_id, aggregate_hash_U, aggregate_hash_D, aggregate_randomness_U, aggregate_randomness_D, client_indexes[receiver_id - 1]
                )

            # start the next round
            self.round_idx += 1
            self.active_clients_first_round = []
            self.active_clients_second_round = []
            self.active_clients_veri_second_round = []
            self.masked_hash_rnd_dict = dict()
            self.aggregator.reset_veri_aggregate_masks()

            # New round starts if this following if-clause does not trigger and allows the clients to again trigger a new handle
            if self.round_idx == self.round_num:
                logging.info("==============================\n SERVER: TRAINING IS FINISHED! \n The final model update has occurred and the final verification aggregate mask was sent.")
                os.makedirs(os.path.dirname(self.modelpath) or ".", exist_ok=True)
                save_model_dict_to_pickle(self.modelpath, self.global_model_params)
                self.barrier()  # wait until every client has verified the last round
                self.finish()



    def handle_message_receive_aggregate_encoded_mask_from_client(self, msg_params):  # sum
        # 8: Receive the aggregate of encoded masks for active clients
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
            # Secure Model Aggregation: y = sum_{U_1} x_i (in F_p); global model = y / |U_1|
            with timer("server: [A.3] decode aggregate mask, recover y = sum x_i", who="server", rnd=self.round_idx):
                aggregate_finite, global_model_params = self.aggregator.aggregate_model_reconstruction(
                    self.active_clients_first_round, self.active_clients_second_round
                )

            # evaluation
            with timer("server: evaluate global model", who="server", rnd=self.round_idx):
                stats = self.aggregator.test_on_server_for_all_clients(self.round_idx)
            if stats is not None:
                self.metrics.append(dict(round=self.round_idx, **stats))

            if self.round_idx == self.round_num - 1:
                logging.info("Last model came in. Server will soon send out the last verification")

            # sampling clients of the next round
            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    # sampling has already been done in data preprocessor
                    client_indexes = [self.round_idx + 1] * self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.round_idx + 1]
            else:
                # sampling clients
                client_indexes = self.aggregator.client_sampling(
                    self.round_idx + 1, self.args.client_num_in_total, self.args.client_num_per_round
                )
            self.client_indexes = client_indexes

            self.global_model_params = global_model_params
            # -> 9: broadcast y = sum x_i (finite field, so that the clients can hash exactly what was aggregated)
            for receiver_id in range(1, self.size):
                self.send_message_sync_model_to_client(
                    receiver_id, aggregate_finite, client_indexes[receiver_id - 1]
                )


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
        logging.debug("Server send_commitment_to_client. receive_id = %d" % receive_id)
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
