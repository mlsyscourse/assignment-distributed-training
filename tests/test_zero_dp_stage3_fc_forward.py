from mpi4py import MPI

import numpy as np
import pytest

from model.zero_dp_stage3 import ZeroDPStage3FCLayer

@pytest.mark.mpi
def test_zero_dp_stage3_fc_forward_divisible_case():
    comm = MPI.COMM_WORLD

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
    out = layer.forward(x)

    expected = np.array(
        [
            [20.0, 16.0, 18.0, 20.0],
            [50.0, 38.0, 36.0, 42.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(actual=out, desired=expected)


@pytest.mark.mpi
def test_zero_dp_stage3_fc_forward_with_padding_case():
    comm = MPI.COMM_WORLD

    layer = ZeroDPStage3FCLayer(
        comm=comm,
        in_dim=3,
        out_dim=8,
        dp_size=4,
        full_w=np.array(
            [
                [59.0, 74.0, 43.0, 93.0, 10.0, 96.0, 4.0, 22.0],
                [85.0, 1.0, 51.0, 63.0, 16.0, 20.0, 1.0, 19.0],
                [34.0, 0.0, 5.0, 7.0, 8.0, 9.0, 3.0, 6.0],
            ],
            dtype=np.float32,
        ),
        full_b=np.array(
            [[48.0, 98.0, 55.0, 5.0, 43.0, 66.0, 93.0, 50.0]], dtype=np.float32
        ),
    )

    x = np.array([[2.0, 1.0, 3.0], [4.0, 0.0, 5.0]], dtype=np.float32)
    out = layer.forward(x)

    expected = np.array(
        [
            [353.0, 247.0, 207.0, 275.0, 103.0, 305.0, 111.0, 131.0],
            [454.0, 394.0, 252.0, 412.0, 123.0, 495.0, 124.0, 168.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(actual=out, desired=expected)
