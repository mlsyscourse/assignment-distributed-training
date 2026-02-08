import argparse
import os

import h5py
import numpy as np
from mpi4py import MPI

from data.data_parallel_preprocess import split_data
from logger import log_args
from mpi_wrapper import Communicator
from model.zero_dp_stage3 import ZeroDPAdam, ZeroDPMLPModel

np.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("--dp_size", type=int, help="data parallel size", default=1)
parser.add_argument("--num_epoch", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=60)
parser.add_argument("--init_lr", type=float, default=0.01)
parser.add_argument("--num_train_samples", type=int, default=None)
parser.add_argument("--num_test_samples", type=int, default=None)


def lr_schedule(init_lr, iter_num, decay=0.9, stage_num=100):
    return init_lr * (decay ** (np.floor(iter_num / stage_num)))


def train_mlp(
    comm,
    x_train,
    y_train,
    x_test,
    y_test,
    model,
    optimizer,
    num_epoch=3,
    batch_size=60,
    init_lr=0.01,
):
    iter_num = 0
    num_examples = x_train.shape[0]
    rank = comm.Get_rank()
    dp_size = comm.Get_size()

    for epoch in range(num_epoch):
        if rank == 0:
            print("*" * 40 + "Training" + "*" * 40)

        for i in range(0, num_examples, batch_size):
            x_batch = (
                x_train[i : i + batch_size]
                if i + batch_size <= num_examples
                else x_train[i:]
            )
            y_batch = (
                y_train[i : i + batch_size]
                if i + batch_size <= num_examples
                else y_train[i:]
            )

            loss, acc = model.forward(x_batch, y_batch)
            model.zero_grad()
            model.backward()

            lr = lr_schedule(init_lr, iter_num, stage_num=100 / dp_size)
            optimizer.lr = lr
            optimizer.step()
            iter_num += 1

            if (iter_num + 1) % 10 == 0 and rank == 0:
                print(
                    f"Epoch:{epoch+1} iter_num:{i}/{num_examples}: Train Loss: {loss}, Train Acc: {acc}, lr_rate: {lr}"
                )

        if rank == 0:
            print("*" * 88)

        # All ranks participate in forward communication during evaluation.
        eval_acc = 0.0
        eval_examples = x_test.shape[0]

        for i in range(0, eval_examples, batch_size):
            x_batch = (
                x_test[i : i + batch_size]
                if i + batch_size <= eval_examples
                else x_test[i:]
            )
            y_batch = (
                y_test[i : i + batch_size]
                if i + batch_size <= eval_examples
                else y_test[i:]
            )
            _, acc = model.forward(x_batch, y_batch)
            eval_acc += acc * x_batch.shape[0]

        if rank == 0:
            print("\n" + "*" * 40 + "Evaluating" + "*" * 40)
            print(f"Test Acc: {eval_acc / x_test.shape[0]}")
            print("*" * 90)


if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    comm = MPI.COMM_WORLD
    comm = Communicator(comm)
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    args = parser.parse_args()
    dp_size = args.dp_size

    assert dp_size == nprocs

    if rank == 0:
        log_args(
            batch_size=args.batch_size,
            init_lr=args.init_lr,
            dp_size=dp_size,
            model_type="zero-dp-stage3",
            optimizer="adam",
        )

    mlp_model = ZeroDPMLPModel(
        comm=comm,
        dp_size=dp_size,
        feature_dim=784,
        hidden_dim=256,
        output_dim=10,
    )
    optimizer = ZeroDPAdam(layers=mlp_model.layers, lr=args.init_lr)

    mnist_data = h5py.File("./data/MNISTdata.hdf5", "r")

    x_train = np.float32(mnist_data["x_train"])
    y_train = np.int32(np.array(mnist_data["y_train"][:, 0]))
    if args.num_train_samples is not None:
        x_train = x_train[: args.num_train_samples]
        y_train = y_train[: args.num_train_samples]

    # ZeRO-DP stage 3 is pure data-parallel for this assignment setup (mp_size=1).
    x_train, y_train = split_data(
        x_train=x_train,
        y_train=y_train,
        mp_size=1,
        dp_size=dp_size,
        rank=rank,
    )

    x_test = np.float32(mnist_data["x_test"][:])
    y_test = np.int32(np.array(mnist_data["y_test"][:, 0]))
    if args.num_test_samples is not None:
        x_test = x_test[: args.num_test_samples]
        y_test = y_test[: args.num_test_samples]
    mnist_data.close()

    np.random.seed(15442)
    idx = np.random.permutation(x_train.shape[0])
    x_train, y_train = x_train[idx], y_train[idx]

    train_mlp(
        comm=comm,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        model=mlp_model,
        optimizer=optimizer,
        num_epoch=args.num_epoch,
        batch_size=int(args.batch_size / dp_size),
        init_lr=args.init_lr,
    )
