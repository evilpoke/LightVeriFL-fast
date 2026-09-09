from mpi4py import MPI

from .LightVeriAggAggregator import LightVeriAggAggregator
from .LightVeriAggClientManager import LightVeriAggClientManager
from .LightVeriAggServerManager import LightVeriAggServerManager
from .LightVeriAggTrainer import LightVeriAggTrainer
from .MyModelTrainer import MyModelTrainer


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_lightveriagg_distributed(
    process_id,
    worker_number,
    device,
    comm,
    model,
    train_data_num,
    train_data_global,
    test_data_global,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    args,
    model_trainer=None,
    preprocessed_sampling_lists=None,
):
    if process_id == 0:
        print("Starting server")
        init_server(
            args,
            device,
            comm,
            process_id,
            worker_number,
            model,
            train_data_num,
            train_data_global,
            test_data_global,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num_dict,
            model_trainer,
            preprocessed_sampling_lists,
        )
    else:
        # for all clients:
        print("Starting client")
        
        init_client(
            args,
            device,
            comm,
            process_id,
            worker_number,
            model,
            train_data_num,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            model_trainer,
        )


def init_server(
    args,
    device,
    comm,
    rank,
    size,
    model,
    train_data_num,
    train_data_global,
    test_data_global,
    train_data_local_dict,
    test_data_local_dict,
    train_data_local_num_dict,
    model_trainer,
    preprocessed_sampling_lists=None,
):
    if model_trainer is None:
        model_trainer = MyModelTrainer(model)
    model_trainer.set_id(-1)

    # aggregator
    worker_num = size - 1
    aggregator = LightVeriAggAggregator(
        train_data_global,
        test_data_global,
        train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        worker_num,
        device,
        args,
        model_trainer,
    )

    # start the distributed training
    backend = args.backend
    if preprocessed_sampling_lists is None:
        server_manager = LightVeriAggServerManager(args, aggregator, comm, rank, size, backend)
    else:
        server_manager = LightVeriAggServerManager(
            args,
            aggregator,
            comm,
            rank,
            size,
            backend,
            is_preprocessed=True,
            preprocessed_client_lists=preprocessed_sampling_lists,
        )
    server_manager.send_init_msg()
    server_manager.run()
    # TODO: time measurements here


def init_client(
    args,
    device,
    comm,
    process_id,
    size,
    model,
    train_data_num,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    model_trainer=None,
):
    client_index = process_id - 1
    if model_trainer is None:
        model_trainer = MyModelTrainer(model)
    model_trainer.set_id(client_index)
    backend = args.backend
    trainer = LightVeriAggTrainer(
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        device,
        args,
        model_trainer,
    )
    client_manager = LightVeriAggClientManager(args, trainer, comm, process_id, size, backend)
    client_manager.run()

