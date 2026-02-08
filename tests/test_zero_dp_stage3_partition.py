import numpy as np

from model.zero_dp_stage3 import ZeroDPStage3FCLayer

def test_partition_flat_tensor_divisible_case():
    layer = ZeroDPStage3FCLayer.__new__(ZeroDPStage3FCLayer)

    tensor = np.array(
        [
            [58.0, 77.0, 56.0, 81.0, 31.0, 45.0, 38.0, 16.0],
            [71.0, 64.0, 60.0, 52.0, 93.0, 43.0, 79.0, 15.0],
        ],
        dtype=np.float32,
    )
    shard, shard_size = layer._partition_flat_tensor(
        tensor=tensor,
        shard_idx=0,
        num_shards=4,
    )

    expected_shard = np.array([58.0, 77.0, 56.0, 81.0], dtype=np.float32)
    assert shard_size == 4
    np.testing.assert_allclose(actual=shard, desired=expected_shard)


def test_partition_flat_tensor_with_padding_case():
    layer = ZeroDPStage3FCLayer.__new__(ZeroDPStage3FCLayer)

    tensor = np.array(
        [
            [
                98.0,
                77.0,
                86.0,
                83.0,
                7.0,
                44.0,
                33.0,
                52.0,
                34.0,
                22.0,
                28.0,
                33.0,
                42.0,
            ],
            [
                97.0,
                11.0,
                97.0,
                88.0,
                39.0,
                74.0,
                58.0,
                98.0,
                15.0,
                53.0,
                64.0,
                73.0,
                39.0,
            ],
            [
                41.0,
                45.0,
                91.0,
                35.0,
                49.0,
                92.0,
                68.0,
                10.0,
                50.0,
                50.0,
                43.0,
                76.0,
                8.0,
            ],
        ],
        dtype=np.float32,
    )
    shard, shard_size = layer._partition_flat_tensor(
        tensor=tensor,
        shard_idx=3,
        num_shards=4,
    )

    expected_shard = np.array(
        [49.0, 92.0, 68.0, 10.0, 50.0, 50.0, 43.0, 76.0, 8.0, 0.0],
        dtype=np.float32,
    )
    assert shard_size == 10
    np.testing.assert_allclose(actual=shard, desired=expected_shard)
