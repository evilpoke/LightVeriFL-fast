import copy
import logging
import random
import time

from fastecdsa.curve import P256
from fedml_api.distributed.utils.EC import LightVeriFL_dec_EC, LightVeriFL_dec_scalar, PI_addEC, curve_n, \
    veri_eval_points
import numpy as np
import torch
#import wandb

from .mpc_function import LCC_decoding_with_points, padded_dimension
from .utils import transform_finite_to_tensor, model_dimension, finite_aggregate_to_averaged_tensor


def aggregate_mask_functioncall_reconstruction(d,N,U,T,p,active_clients, aggregate_encoded_mask_dict):
    """
    Recover the aggregate-mask via decoding (from the first U of the active clients)
    d is the model dimension, the masks have padded_dimension(d, U, T) entries
    """
    
    logging.debug("d = {}, N = {}, U = {}, T = {}, p = {}".format(d, N, U, T, p))
    active_clients = list(active_clients)[:U]
    assert len(active_clients) == U, "LightSecAgg needs the aggregated encoded masks of U = %d clients" % U
    d_pad = padded_dimension(d, U, T)
    alpha_s = np.array(range(N)) + 1
    beta_s = np.array(range(U)) + (N+1)
    logging.info("Server starts the reconstruction of aggregate_mask")
    aggregate_encoded_mask_buffer = np.zeros((U, d_pad // (U - T)), dtype="int64")
    for i, client_idx in enumerate(active_clients):
        aggregate_encoded_mask_buffer[i, :] = aggregate_encoded_mask_dict[client_idx]
    eval_points = alpha_s[active_clients]
    aggregate_mask = LCC_decoding_with_points(aggregate_encoded_mask_buffer, eval_points, beta_s, p)
    logging.info("Server finish the reconstruction of aggregate_mask via LCC decoding")
    aggregate_mask = np.reshape(aggregate_mask, (U * (d_pad // (U - T)), 1))
    aggregate_mask = aggregate_mask[0:d]
    return aggregate_mask

class LightVeriAggAggregator(object):
    def __init__(
        self,
        train_global,
        test_global,
        all_train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        worker_num,
        device,
        args,
        model_trainer,
    ):
        self.trainer = model_trainer

        self.args = args
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set()
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.T = self.worker_num // 2
        self.device = device
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.aggregate_encoded_mask_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        self.flag_client_aggregate_encoded_mask_uploaded_dict = dict()

        self.total_dimension = None
        self.dimensions = []
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
            self.flag_client_aggregate_encoded_mask_uploaded_dict[idx] = False

        self.targeted_number_active_clients = args.targeted_number_active_clients
        self.privacy_guarantee = args.privacy_guarantee
        self.prime_number = args.prime_number
        self.precision_parameter = args.precision_parameter

        # verification
        self.aggregate_veri_encoded_mask_U_dict = dict()
        self.flag_client_aggregate_veri_encoded_mask_uploaded_U_dict = dict()
        self.aggregate_veri_encoded_mask_D_dict = dict()
        self.flag_client_aggregate_veri_encoded_mask_uploaded_D_dict = dict()


    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def get_model_dimension(self, weights):
        self.dimensions, self.total_dimension = model_dimension(weights)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def add_local_veri_aggregate_encoded_mask_U(self, index, aggregate_veri_encoded_mask_U):
        logging.info("add_local_veri_aggregate_encoded_mask_U: index = %d" % index)
        self.aggregate_veri_encoded_mask_U_dict[index] = aggregate_veri_encoded_mask_U
        self.flag_client_aggregate_veri_encoded_mask_uploaded_U_dict[index] = True


    def add_local_veri_aggregate_encoded_mask_D(self, index, aggregate_veri_encoded_mask_D):
        logging.info("add_local_veri_aggregate_encoded_mask_D: index = %d" % index)
        self.aggregate_veri_encoded_mask_D_dict[index] = aggregate_veri_encoded_mask_D
        self.flag_client_aggregate_veri_encoded_mask_uploaded_D_dict[index] = True


    def add_local_aggregate_encoded_mask(self, index, aggregate_encoded_mask):
        logging.info("add_aggregate_encoded_mask index = %d" % index)
        self.aggregate_encoded_mask_dict[index] = aggregate_encoded_mask
        self.flag_client_aggregate_encoded_mask_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def reset_veri_aggregate_masks(self):
        self.aggregate_veri_encoded_mask_U_dict = dict()
        self.flag_client_aggregate_veri_encoded_mask_uploaded_U_dict = dict()
        self.aggregate_veri_encoded_mask_D_dict = dict()
        self.flag_client_aggregate_veri_encoded_mask_uploaded_D_dict = dict()

    def check_whether_all_aggregate_encoded_mask_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_aggregate_encoded_mask_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_aggregate_encoded_mask_uploaded_dict[idx] = False
        return True

    def aggregate_veriaggmask_reconstruction_U(self, active_clients_veri):
        """
        Recover the aggregate veri mask (sum zh_i, sum zr_i) of the surviving users U = U_1
        """
        return self.decode_veri_aggregate(self.aggregate_veri_encoded_mask_U_dict, active_clients_veri)

    def aggregate_veriaggmask_reconstruction_D(self, active_clients_veri):
        """
        Recover the aggregate veri mask (sum zh_i, sum zr_i) of the dropped users D = [N] \\ U_1
        """
        return self.decode_veri_aggregate(self.aggregate_veri_encoded_mask_D_dict, active_clients_veri)

    def decode_veri_aggregate(self, aggregate_veri_encoded_mask_dict, active_clients_veri):
        """
        One-shot MDS decoding from T+1 aggregated encoded verification masks:
        EC part (mask of the hashes) and integer part (mask of the El Gamal randomness)
        """
        N = self.worker_num
        T = self.privacy_guarantee
        evalpoints_in, evalpoints_out = veri_eval_points(N, T)
        responders = sorted(active_clients_veri)[: T + 1]
        assert len(responders) == T + 1, "need T+1 = %d aggregated encoded verification masks" % (T + 1)
        shares_hash = [aggregate_veri_encoded_mask_dict[i][0] for i in responders]
        shares_rnd = [aggregate_veri_encoded_mask_dict[i][1] for i in responders]
        points = [evalpoints_out[i] for i in responders]
        dec_zh = LightVeriFL_dec_EC(shares_hash, evalpoints_in, points, P256)
        dec_zr = LightVeriFL_dec_scalar(shares_rnd, evalpoints_in, points, curve_n)
        return dec_zh, dec_zr


    def aggregate_mask_reconstruction(self, active_clients):
        """
        Recover the aggregate-mask via decoding. Calls aggregate_mask_functioncall_reconstruction
        """
        
        d = self.total_dimension
        N = self.worker_num
        U = self.targeted_number_active_clients
        T = self.privacy_guarantee
        p = self.prime_number

        return aggregate_mask_functioncall_reconstruction(d,N,U,T,p,active_clients, self.aggregate_encoded_mask_dict)
        
    def aggregate_model_reconstruction(self, active_clients_first_round, active_clients_second_round):
        """
        returns y = sum_{i in U_1} x_i in F_p (dict of int64 arrays) and the new global model y / |U_1| (tensors)
        """
        start_time = time.time()
        aggregate_mask = self.aggregate_mask_reconstruction(active_clients_second_round)
        p = self.prime_number
        q_bits = self.precision_parameter
        logging.info("Server starts the reconstruction of aggregate_model")
        first = self.model_dict[active_clients_first_round[0]]
        aggregate_finite = dict()
        pos = 0
        for j, k in enumerate(first):
            acc = np.zeros(np.shape(first[k]), dtype=np.int64)
            for i, client_idx in enumerate(active_clients_first_round):
                acc = np.mod(acc + np.asarray(self.model_dict[client_idx][k], dtype=np.int64), p)
            cur_shape = np.shape(acc)
            d = self.dimensions[j]
            cur_mask = np.array(aggregate_mask[pos : pos + d, :])
            cur_mask = np.reshape(cur_mask, cur_shape)

            # Cancel out the aggregate-mask to recover the aggregate-model
            aggregate_finite[k] = np.mod(acc - cur_mask, p)
            pos += d

        # Convert the model from finite to real and average ("do the avg after transform")
        logging.info("Server converts the aggregate_model from finite to tensor")
        averaged_params = finite_aggregate_to_averaged_tensor(aggregate_finite, p, q_bits, len(active_clients_first_round))

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        end_time = time.time()
        logging.info("aggregate time cost: %.3f s" % (end_time - start_time))
        return aggregate_finite, averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        else:
            return self.test_global

    def test_on_server_for_all_clients(self, round_idx):
        if self.trainer.test_on_the_server(
            self.train_data_local_dict, self.test_data_local_dict, self.device, self.args
        ):
            return

        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################test_on_server_for_all_clients : {}".format(round_idx))
            # train_num_samples = []
            # train_tot_corrects = []
            # train_losses = []
            # for client_idx in range(self.args.client_num_in_total):
            #     # train data
            #     metrics = self.trainer.test(self.train_data_local_dict[client_idx], self.device, self.args)
            #     train_tot_correct, train_num_sample, train_loss = (
            #         metrics["test_correct"],
            #         metrics["test_total"],
            #         metrics["test_loss"],
            #     )
            #     train_tot_corrects.append(copy.deepcopy(train_tot_correct))
            #     train_num_samples.append(copy.deepcopy(train_num_sample))
            #     train_losses.append(copy.deepcopy(train_loss))
            #
            #     """
            #     Note: CI environment is CPU-based computing.
            #     The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            #     """
            #     if self.args.ci == 1:
            #         break
            #
            # # test on training dataset
            # train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            # train_loss = sum(train_losses) / sum(train_num_samples)
            # wandb.log({"Train/Acc": train_acc, "round": round_idx})
            # wandb.log({"Train/Loss": train_loss, "round": round_idx})
            # stats = {"training_acc": train_acc, "training_loss": train_loss}
            # logging.info(stats)

            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            if round_idx == self.args.comm_round - 1:
                metrics = self.trainer.test(self.test_global, self.device, self.args)
            else:
                metrics = self.trainer.test(self.val_global, self.device, self.args)

            test_tot_correct, test_num_sample, test_loss = (
                metrics["test_correct"],
                metrics["test_total"],
                metrics["test_loss"],
            )
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            #wandb.log({"Test/Acc": test_acc, "round": round_idx})
            #wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {"test_acc": test_acc, "test_loss": test_loss}
            logging.info(stats)
            return stats
