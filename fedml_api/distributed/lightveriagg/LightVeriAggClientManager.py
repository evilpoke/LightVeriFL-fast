import copy
import logging
import os
import pickle
import sys

from fastecdsa.curve import P256
from fedml_api.distributed.utils.EC import LightVeriFL_enc_EC, LightVeriFL_enc_scalar, PI_addEC, \
    aggregate_el_gamal_commitments, compute_elgamal_commitment_to_hash_randomness, curve_n, ec_add, ec_mul, \
    elgamal_commitments_equal, generate_point_EC, get_hash_bases, hash_vector, random_scalar, veri_eval_points
from fedml_api.distributed.utils.bench import timer
import numpy as np
import secrets
import time
from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.client.client_manager import ClientManager
from .message_define import MyMessage
from .utils import transform_list_to_tensor, transform_tensor_to_finite, flatten_finite_model, \
    finite_aggregate_to_averaged_tensor
from .utils import model_dimension
from .mpc_function import model_masking, mask_encoding, compute_aggregate_encoded_mask, padded_dimension, \
    random_field_array, verification_dropouts


class VerificationError(Exception):
    """Raised (single-process mode) / logged (MPI) when a client detects a forged aggregation."""


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
        self.dimensions = []
        self.total_dimension = None                     # is initialized with the model init
        for idx in range(self.worker_num):
            self.flag_encoded_mask_dict[idx] = False


        # new added parameters in main file
        self.targeted_number_active_clients = args.targeted_number_active_clients
        self.privacy_guarantee = args.privacy_guarantee
        self.prime_number = args.prime_number
        self.precision_parameter = args.precision_parameter

        # verification
        self.N = self.size - 1
        self.client_idx = self.rank - 1                 # 0-based index of this client (= position in U_1 lists)
        self.hash_bases = None
        self.stored_model = None                        # x_i in F_p (dict) of the current round
        self.hz = None                                  # masked hash        h_i + zh_i   (EC point)
        self.rz = None                                  # masked randomness  r_i + zr_i   (in Z_n)
        self.local_veri_mask = None                     # (zh_i, zr_i): EC point mask of h_i, integer mask of r_i
        self.all_commmitments = dict()                  # client index -> El Gamal commitment (c1, c2)
        self.encoded_veri_mask_dict = dict()
        self.flag_encoded_veri_mask_dict = dict()
        for idx in range(self.worker_num):
            self.flag_encoded_veri_mask_dict[idx] = False
        self.local_sample_num = -1
        self.active_clients_first_round = []
        self.received_aggregate = None                  # y = sum_{U_1} x_j in F_p (dict)
        self.uploaded_this_round = False
        self.veri_dropout_this_round = False

        # amortized verification: users check the aggregations of the last L rounds at once
        self.check_second_veri_every_amount = max(1, int(getattr(args, "veri_batch_size", self.num_rounds)))
        self.veri_alpha_bits = int(getattr(args, "veri_alpha_bits", 64))
        self.veri_dropout_rate = float(getattr(args, "veri_dropout_rate", 0.0))
        self.veri_dropout_seed = int(getattr(args, "veri_dropout_seed", 0))
        self.batch_buffer = []                          # [(round, alpha_l, y(l) signed, h^l)]
        self.verification_log = []                      # per round: what this client verified
        self.verification_failed = False


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
        # 13: commitment c_j of client j (forwarded by the server)
        commitment_from_client = msg_params.get(MyMessage.MSG_ARG_KEY_COMMITMENT)
        client_id = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_ID)
        self.add_commitment(client_id - 1, commitment_from_client)
        self.upload_masked_inputs_if_ready()

    def upload_masked_inputs_if_ready(self):
        # Once every commitment has been exchanged (and the own one is computed):
        #   -> 5 upload \tilde h_i, \tilde r_i      -> 6 upload \tilde x_i = x_i + z_i
        if self.uploaded_this_round or self.stored_model is None or not self.check_whether_all_commitments_receive():
            return
        self.uploaded_this_round = True
        logging.info("client %d has received all commitments" % self.get_sender_id())

        # now uploading masked hash and masked randomness
        self.send_veri_masked_hash_randomness()

        # Mask the local model
        with timer("client: [A.2] mask model x_i + z_i", who=self.rank, rnd=self.round_idx):
            weights_finite = copy.deepcopy(self.stored_model)
            masked_weights = model_masking(weights_finite, self.dimensions, self.local_mask, self.prime_number)

        if self.local_sample_num == -1:
            raise Exception("No training has happened yet.")

        self.send_model_to_server(0, masked_weights, self.local_sample_num)


    def handle_message_init(self, msg_params):
        # gets initial global model from the federator
        logging.info("client %d handle_message_init from server." % self.get_sender_id())

        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        if self.args.is_mobile == 1:
            global_model_params = transform_list_to_tensor(global_model_params)

        # inits new model by these new global params
        self.dimensions, self.total_dimension = model_dimension(global_model_params)
        # public parameters of the hash (deterministic, identical at every party, computed once per process)
        t0 = time.time()
        self.hash_bases = get_hash_bases(self.total_dimension)
        logging.info("client %d: hash bases g_1..g_%d ready (%.2f s, one-time setup)"
                     % (self.get_sender_id(), self.total_dimension, time.time() - t0))

        # updates the model
        self.trainer.update_model(global_model_params)
        self.trainer.update_dataset(int(client_index))

        self.round_idx = 0
        self.__offline()


    def handle_message_receive_encoded_mask_from_server(self, msg_params):
        # 12: [\tilde zv_j]_i and [\tilde z_j]_i of client j
        encoded_mask = msg_params.get(MyMessage.MSG_ARG_KEY_ENCODED_MASK)
        encoded_veri_mask = msg_params.get(MyMessage.MSG_ARG_KEY_VERI_ENCODED_MASK)

        client_id = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_ID)
        logging.debug("Client %d receive encoded masks from Client %d" % (self.get_sender_id(), client_id))
        self.add_encoded_mask(client_id - 1, encoded_mask)
        self.add_encoded_veri_mask(client_id-1, encoded_veri_mask)
        b_all_received = self.check_whether_all_encoded_mask_receive()
        if b_all_received:
            # [Training]
            with timer("client: [A.2] local training", who=self.rank, rnd=self.round_idx):
                x = self.__train()

            # store x (in F_p)
            self.stored_model = x

            # [Veri Commit]: h_i = H(x_i), r_i <- Z_n, c_i = (r_i G, h_i + r_i H)
            with timer("client: [A.2] hash h_i = H(x_i)", who=self.rank, rnd=self.round_idx):
                h = hash_vector(flatten_finite_model(x, self.prime_number), self.hash_bases)

            with timer("client: [A.2] El Gamal commitment c_i", who=self.rank, rnd=self.round_idx):
                randomness = random_scalar()
                commitment = compute_elgamal_commitment_to_hash_randomness(h, randomness)

            # masking the hash (EC point mask) and the randomness (separate integer mask in Z_n)
            with timer("client: [A.2] mask h_i and r_i", who=self.rank, rnd=self.round_idx):
                zh, zr = self.local_veri_mask
                self.hz = ec_add(h, zh)
                self.rz = (randomness + zr) % curve_n

            self.add_commitment(self.client_idx, commitment)

            # send commitment to server for broadcast
            self.send_veri_commitment(commitment)

            # all other commitments might already be here
            self.upload_masked_inputs_if_ready()



    def handle_message_hash_randomess_aggreg(self, msg_params):
        # ----------------
        # 11: sum_U h, sum_D h, sum_U r, sum_D r (U = U_1, D = [N] \ U_1) reconstructed by the server
        # - checking the integrity of the reconstruction (commitments) and of the aggregation (batch check)
        # - Starting the new round, beginning with
        #    [Separate verification mask]
        #       - client generates zv_i , generates [nv_i]_js , encode to [\tilde zv_i]_j
        #       -> 3 (client sends [\tilde zv_i]_j to other clients via server)            MSG_TYPE_C2S_SEND_VERI_ENCODED_MASK_TO_SERVER
        # ----------------
        logging.info("client %d handle_message_hash_randomess_aggreg." % self.get_sender_id())
        aggregate_hash_U = msg_params.get(MyMessage.MSG_ARG_KEY_AGGREGATE_HASH_FOR_U)
        aggregate_hash_D = msg_params.get(MyMessage.MSG_ARG_KEY_AGGREGATE_HASH_FOR_D)
        aggregate_rnd_U = msg_params.get(MyMessage.MSG_ARG_KEY_AGGREGATE_RND_FOR_U)
        aggregate_rnd_D = msg_params.get(MyMessage.MSG_ARG_KEY_AGGREGATE_RND_FOR_D)

        record = {"round": self.round_idx, "client": self.client_idx, "dropped_in_verification": self.veri_dropout_this_round}
        if not self.veri_dropout_this_round:
            # [Round V.0] COM(sum h_i, sum r_i) == sum c_i   (separately for U and D)
            with timer("client: [V.0] verify commitments", who=self.rank, rnd=self.round_idx):
                surviving = sorted(self.active_clients_first_round)
                non_surviving = [i for i in range(self.N) if i not in surviving and i in self.all_commmitments]
                commitments_summed_U = aggregate_el_gamal_commitments([self.all_commmitments[i] for i in surviving])
                commitments_summed_D = aggregate_el_gamal_commitments([self.all_commmitments[i] for i in non_surviving])
                ok_U = elgamal_commitments_equal(
                    compute_elgamal_commitment_to_hash_randomness(aggregate_hash_U, aggregate_rnd_U), commitments_summed_U)
                ok_D = elgamal_commitments_equal(
                    compute_elgamal_commitment_to_hash_randomness(aggregate_hash_D, aggregate_rnd_D), commitments_summed_D)
            record["commitments_ok"] = bool(ok_U and ok_D)
            if not (ok_U and ok_D):
                self.verification_log.append(record)
                self.forged("Forged reconstruction of the aggregate hash/randomness at the server "
                            "(commitment check failed, U: %s, D: %s)" % (ok_U, ok_D))
                return
            logging.info("client %d confirmed correct commit aggregation" % self.get_sender_id())

            # store y(l) and h^l = sum_{i in U_1} h_i for the amortized check with a private random alpha_l
            alpha_l = 1 if self.veri_alpha_bits <= 0 else 1 + secrets.randbits(self.veri_alpha_bits)
            y_signed = flatten_finite_model(self.received_aggregate, self.prime_number)
            self.batch_buffer.append((self.round_idx, alpha_l, y_signed, aggregate_hash_U))

            last_round = self.round_idx == self.num_rounds - 1
            if (self.round_idx + 1) % self.check_second_veri_every_amount == 0 or last_round:
                # [Round V.1] batch checking: h(sum_l alpha_l y(l)) == sum_l alpha_l h^l
                with timer("client: [V.1] batch check h(sum a_l y(l)) == sum a_l h^l", who=self.rank, rnd=self.round_idx):
                    combined_model = sum(y.astype(object) * a for (_, a, y, _) in self.batch_buffer)
                    h_agg = hash_vector(combined_model, self.hash_bases)
                    h_blindly_added = PI_addEC([ec_mul(a, h_l) for (_, a, _, h_l) in self.batch_buffer])
                    ok = h_agg == h_blindly_added
                checked_rounds = [r for (r, _, _, _) in self.batch_buffer]
                record["batch_checked_rounds"] = checked_rounds
                record["batch_ok"] = bool(ok)
                self.batch_buffer = []
                if not ok:
                    self.verification_log.append(record)
                    self.forged("Forged aggregation: hash of the aggregate models of rounds %s does not match" % checked_rounds)
                    return
                logging.info("client %d confirmed correct aggregation of rounds %s" % (self.get_sender_id(), checked_rounds))
        else:
            logging.info("client %d dropped in the verification of round %d (simulated)" % (self.get_sender_id(), self.round_idx))
        self.verification_log.append(record)

        # Restarting to the next round.
        self.round_idx += 1
        self.reset_round_state()
        if self.round_idx == self.num_rounds:
            logging.info("client %d has finished the training and the verification!" % self.get_sender_id())
            self.finish_protocol()
            return
        logging.info("client %d starts a new FL round..." % self.get_sender_id())
        self.__offline()


    def forged(self, reason):
        # the client regards the result as forged and stops (as the former exit(1))
        logging.error("client %d: %s" % (self.get_sender_id(), reason))
        self.verification_failed = True
        if self.backend == "MPI":
            self.finish()
        raise VerificationError(reason)

    def finish_protocol(self):
        if self.backend == "MPI":
            from mpi4py import MPI
            MPI.COMM_WORLD.Barrier()  # the server waits here until every client has verified the last round
            self.finish()


    def reset_round_state(self):
        self.stored_model = None
        self.hz = None
        self.rz = None
        self.all_commmitments = dict()
        self.uploaded_this_round = False
        self.received_aggregate = None
        self.veri_dropout_this_round = False


    def handle_message_receive_model_from_server(self, msg_params):
        # ----------------
        # 9: y = sum_{U_1} x_j (in F_p) from the server
        # ----------------
        logging.info("client %d handle_message_receive_model_from_server." % self.get_sender_id())
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        with timer("client: update local model with y / |U_1|", who=self.rank, rnd=self.round_idx):
            self.received_aggregate = {k: np.asarray(v, dtype=np.int64) for k, v in model_params.items()}
            global_model_params = finite_aggregate_to_averaged_tensor(
                self.received_aggregate, self.prime_number, self.precision_parameter, len(self.active_clients_first_round))

            # training update the local model
            self.trainer.update_model(global_model_params)
            self.trainer.update_dataset(int(client_index))

        # U = U_1 (from lightsecagg); D = [N] \ U_1
        dropped = verification_dropouts(self.round_idx, self.active_clients_first_round, self.veri_dropout_rate,
                                        self.privacy_guarantee, self.veri_dropout_seed)
        self.veri_dropout_this_round = self.client_idx in dropped
        if self.veri_dropout_this_round:
            return

        # sending the aggregated veri masks for resolving
        with timer("client: [V.0] aggregate encoded verification masks", who=self.rank, rnd=self.round_idx):
            surviving = sorted(self.active_clients_first_round)
            non_surviving = [i for i in range(self.N) if i not in surviving and i in self.encoded_veri_mask_dict]
            aggregate_veri_mask_U = self.aggregate_encoded_veri_masks(surviving)
            aggregate_veri_mask_D = self.aggregate_encoded_veri_masks(non_surviving)

        self.send_veri_aggregate_mask(0, aggregate_veri_mask_U, aggregate_veri_mask_D)

    def aggregate_encoded_veri_masks(self, clients):
        # sum_j [\tilde zh_j]_i (EC points) and sum_j [\tilde zr_j]_i (integers mod n)
        return (PI_addEC([self.encoded_veri_mask_dict[j][0] for j in clients]),
                sum(int(self.encoded_veri_mask_dict[j][1]) for j in clients) % curve_n)


    def handle_message_receive_active_from_server(self, msg_params):
        # 7: U_1 -> 8: sum_{j in U_1} [\tilde z_j]_i
        # Receive the set of active client id in first round
        active_clients_first_round = msg_params.get(MyMessage.MSG_ARG_KEY_ACTIVE_CLIENTS)
        self.active_clients_first_round = list(active_clients_first_round)
        logging.info(
            "Client %d receive active_clients in the first round = %s"
            % (self.get_sender_id(), active_clients_first_round)
        )
        # Compute the aggregate of encoded masks for the active clients
        with timer("client: [A.3] aggregate encoded masks (LightSecAgg)", who=self.rank, rnd=self.round_idx):
            p = self.prime_number
            aggregate_encoded_mask = compute_aggregate_encoded_mask(self.encoded_mask_dict, p, active_clients_first_round)

        # Send the aggregate of encoded mask to server
        self.send_aggregate_encoded_mask_to_server(0, aggregate_encoded_mask)

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def send_veri_commitment(self, commitment):
        # one upload; the server forwards c_i to every other client
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
        message.add_params(MyMessage.MSG_ARG_KEY_MASKED_HASH_U, self.hz)
        message.add_params(MyMessage.MSG_ARG_KEY_MASKED_RND_U,  self.rz)
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
            if idx not in self.all_commmitments:
                return False
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

                self.send_veri_encoded_mask_to_server(receive_id, encoded_veri_mask) # 3: MSG_TYPE_C2S_SEND_VERI_ENCODED_MASK_TO_SERVER
                self.send_encoded_mask_to_server(receive_id, encoded_mask)           # 2: MSG_TYPE_C2S_SEND_ENCODED_MASK_TO_SERVER

            else:
                self.add_encoded_mask(receive_id - 1, encoded_mask)
                self.add_encoded_veri_mask(receive_id - 1, encoded_veri_mask)

    def __offline(self):
        # - Creating a local mask z and zv = (zh, zr)
        # - Sharding z and zv
        # - sending z_ij and zv_ij to the respective client

        logging.info("#######Client %d offline encoding round_id = %d######" % (self.get_sender_id(), self.round_idx))

        d = self.total_dimension
        N = self.size - 1
        U = self.targeted_number_active_clients
        T = self.privacy_guarantee
        p = self.prime_number
        logging.debug("d = {}, N = {}, U = {}, T = {}, p = {}".format(d, N, U, T, p))

        # [LightSeg mask] z_i in F_p^d (padded to a multiple of U-T), split into U-T pieces, T blindings, MDS encoded
        with timer("client: [A.1] LightSecAgg mask z_i: generate + encode", who=self.rank, rnd=self.round_idx):
            d_pad = padded_dimension(d, U, T)
            self.local_mask = random_field_array(p, (d_pad, 1))                 # mask z_i for client i
            encoded_mask_set = mask_encoding(d_pad, N, U, T, p, self.local_mask)

        # [Separate verification mask] zv_i = (zh_i, zr_i):
        #   zh_i : random EC point, masks the hash h_i         -> (N, T+1) MDS code over the curve (LightVeriFL)
        #   zr_i : random integer in Z_n, masks the El Gamal randomness r_i -> the same MDS code over Z_n
        with timer("client: [A.1] verification masks zh_i, zr_i: generate + encode", who=self.rank, rnd=self.round_idx):
            zh = generate_point_EC()
            nh_array = [generate_point_EC() for _ in range(T)]
            zr = random_scalar()
            nr_array = [random_scalar() for _ in range(T)]
            self.local_veri_mask = (zh, zr)
            evalpoints_in, evalpoints_out = veri_eval_points(N, T)
            encoded_zh = LightVeriFL_enc_EC(zh, nh_array, evalpoints_in, evalpoints_out, P256)
            encoded_zr = LightVeriFL_enc_scalar(zr, nr_array, evalpoints_in, evalpoints_out, curve_n)
            encoded_veri_mask_set = list(zip(encoded_zh, encoded_zr))

        # Send the encoded masks to other clients (via server)
        self.encoded_mask_sharing(encoded_mask_set, encoded_veri_mask_set)

    def __train(self):
        logging.info("Client %d #######training########### round_id = %d" % (self.get_sender_id(), self.round_idx))
        weights, local_sample_num = self.trainer.train(self.round_idx)
        self.local_sample_num = local_sample_num
        # Convert the model from real to finite
        p = self.prime_number
        q_bits = self.precision_parameter
        weights_finite = transform_tensor_to_finite(weights, p, q_bits)

        return weights_finite
