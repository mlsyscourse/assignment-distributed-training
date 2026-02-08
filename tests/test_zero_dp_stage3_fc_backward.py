from mpi4py import MPI

import numpy as np
import pytest

# from model.zero_dp_stage3 import ZeroDPStage3FCLayer
from reference_solution.zero_dp_stage3 import ZeroDPStage3FCLayer


@pytest.mark.mpi
def test_zero_dp_stage3_fc_backward_divisible_case():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    layer = ZeroDPStage3FCLayer(
        comm=comm,
        in_dim=2,
        out_dim=4,
        dp_size=4,
        full_w=np.array(
            [[11.0, 6.0, 2.0, 3.0], [4.0, 5.0, 7.0, 8.0]], dtype=np.float32
        ),
        full_b=np.array([[1.0, 0.0, 2.0, 1.0]], dtype=np.float32),
    )

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    _ = layer.forward(x)

    output_grad = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
    )
    grad_x = layer.backward(output_grad)[0]

    expected_grad_x = np.array([[41.0, 67.0], [129.0, 163.0]], dtype=np.float32)
    expected_grad_w_shard = {
        0: np.array([64.0, 80.0], dtype=np.float32),
        1: np.array([96.0, 112.0], dtype=np.float32),
        2: np.array([88.0, 112.0], dtype=np.float32),
        3: np.array([136.0, 160.0], dtype=np.float32),
    }
    expected_grad_b_shard = {
        0: np.array([24.0], dtype=np.float32),
        1: np.array([32.0], dtype=np.float32),
        2: np.array([40.0], dtype=np.float32),
        3: np.array([48.0], dtype=np.float32),
    }

    np.testing.assert_allclose(actual=grad_x, desired=expected_grad_x)
    np.testing.assert_allclose(
        actual=layer.grad_w_shard, desired=expected_grad_w_shard[rank]
    )
    np.testing.assert_allclose(
        actual=layer.grad_b_shard, desired=expected_grad_b_shard[rank]
    )


@pytest.mark.mpi
def test_zero_dp_stage3_fc_backward_with_padding_case():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    layer = ZeroDPStage3FCLayer(
        comm=comm,
        in_dim=2,
        out_dim=3,
        dp_size=4,
        full_w=np.array([[2.0, 1.0, 3.0], [4.0, 0.0, 5.0]], dtype=np.float32),
        full_b=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
    )

    x = np.array([[2.0, 1.0]], dtype=np.float32)
    _ = layer.forward(x)

    output_grad = np.array([[3.0, 4.0, 5.0]], dtype=np.float32)
    grad_x = layer.backward(output_grad)[0]

    expected_grad_x = np.array([[25.0, 37.0]], dtype=np.float32)
    expected_grad_w_shard = {
        0: np.array([24.0, 32.0], dtype=np.float32),
        1: np.array([40.0, 12.0], dtype=np.float32),
        2: np.array([16.0, 20.0], dtype=np.float32),
        3: np.array([0.0, 0.0], dtype=np.float32),
    }
    expected_grad_b_shard = {
        0: np.array([12.0], dtype=np.float32),
        1: np.array([16.0], dtype=np.float32),
        2: np.array([20.0], dtype=np.float32),
        3: np.array([0.0], dtype=np.float32),
    }

    np.testing.assert_allclose(actual=grad_x, desired=expected_grad_x)
    np.testing.assert_allclose(
        actual=layer.grad_w_shard, desired=expected_grad_w_shard[rank]
    )
    np.testing.assert_allclose(
        actual=layer.grad_b_shard, desired=expected_grad_b_shard[rank]
    )
