import numpy as np
from mpi4py import MPI


def get_info(
    comm,
    rank: int,
    mp_size: int,
    dp_size: int,
    is_fc1: bool,
    is_megatron_mp: bool,
    in_dim: int,
    out_dim: int,
):
    """The function that prepare necessary information for parallel training.

    Parameters
    ----------
        comm : Communicator
            the global mpi communicator

        rank : int
            the corresponding rank of the process

        mp_size : int
            Model Parallel size

        dp_size : int
            Data Parallel size

        is_fc1 : int
            A boolean indicating whether the current layer is the first layer or not

        is_megatron_mp : boolean
            A boolean indicating whether we are using Megatron-style Model Parallel or not

        in_dim : int
            An integer corresponds to the original input feature dimension

        out_dim : int
            An integer corresponds to the original output feature dimension

    Returns
    -------
        mp_idx : int
            An integer corresponds to model parallel communication index

        dp_idx : int
            An integer corresponds to data parallel communication index

        mp_comm : Communicator
            The Model Parallel communicator after split

        dp_comm : Communicator
            The Data Parallel communicator after split

        part_in_dim : int
            An integer corresponds to the input feature dimension after specific parallelism

        part_out_dim : int
            An integer corresponds to the output feature dimension after specific parallelism
    """

    # Get the mp_idx, dp_idx from rank, mp_size and dp_size (you may not need to use all three of them)

    mp_idx = rank % mp_size
    dp_idx = rank // mp_size

    # Get the model/data parallel communication groups
    # the model/data parallel communication group is required to apply mpi operations within the scope of the group
    # Hint: try to figure out the relationship between the mp_idx, dp_idx with the mp/dp communication group
    #       and use the comm.Split() function to get the corresponding group.

    mp_comm = comm.Split(color=dp_idx, key=mp_idx)
    dp_comm = comm.Split(color=mp_idx, key=dp_idx)

    # Derive the part_in_dim and part_out_dim depend on is_fc1 and is_megatron_mp

    if is_megatron_mp:
        if is_fc1:
            part_in_dim = in_dim
            part_out_dim = out_dim // mp_size
        else:
            part_in_dim = in_dim // mp_size
            part_out_dim = out_dim
    else:
        part_in_dim = in_dim
        part_out_dim = out_dim // mp_size


    return mp_idx, dp_idx, mp_comm, dp_comm, part_in_dim, part_out_dim


def naive_collect_forward_input(
    x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """The function for collecting layer fc2's inputs across different nodes with naive model parallelism

    Parameters
    ----------
        x : np.ndarray
            layer input for a single node of shape (batch_size, part_in_dim)

        mp_comm : Communicator
            The Model Parallel communicator

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_x : np.ndarray
            collected layer inputs across different nodes of shape (batch_size, in_dim)

    """

    

    # Note: you may want to ensure that the source variable and destination variable in your mpi func call should
    #       have the same data type, otherwise you will not collect the correct value.

    # Hint: Try to figure out the way MPI calls deal with the destination memory layout for 2d matrix transfer, this might
    #       might not align with your expected layout. In order to get the correct layout, you may wish to use some NumPy
    #       functions (np.split and np.concatenate might be helpful).

    batch_size, part_in_dim = x.shape
    collected = np.empty((batch_size * mp_size, part_in_dim), dtype=x.dtype)
    mp_comm.Allgather(x, collected)
    collected_x = np.concatenate(np.split(collected, mp_size, axis=0), axis=1)
    return collected_x


def naive_collect_forward_output(
    out: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """The function for collecting layer fc2's outputs across different nodes with naive model parallelism

    Parameters
    ----------
        out : np.ndarray
            layer output for a single node of shape (batch_size, part_out_dim)

        mp_comm : Communicator
            The Model Parallel communicator

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_out : np.ndarray
            collected layer outputs across different nodes of shape (batch_size, out_dim)

    """

    

    # Hint: you might have just implemented something similar ^-^
    batch_size, part_out_dim = out.shape
    collected = np.empty((batch_size * mp_size, part_out_dim), dtype=out.dtype)
    mp_comm.Allgather(out, collected)
    collected_out = np.concatenate(np.split(collected, mp_size, axis=0), axis=1)
    return collected_out


def megatron_collect_forward_input(
    x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """The function for collecting layer fc2's inputs across different nodes with megatron-style model parallelism

    Parameters
    ----------
        x : np.ndarray
            layer input for a single node of shape (batch_size, part_in_dim)

        mp_comm : Communicator
            The Model Parallel communicator

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_x : np.ndarray
            collected layer inputs across different nodes of shape (batch_size, in_dim)

    """

    

    # Hint: you don't need all the input parameters to get the collected_x

    return x


def megatron_collect_forward_output(
    out: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """The function for collecting layer fc2's outputs across different nodes with megatron-style model parallelism

    Parameters
    ----------
        out : np.ndarray
            layer output for a single node of shape (batch_size, part_out_dim)

        mp_comm : Communicator
            The Model Parallel communicator

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_out : np.ndarray
            collected layer outputs across different nodes of shape (batch_size, out_dim)

    """

    

    # Hint: try to work through a toy forward example for megatron-style model parallel to figure out the
    #       the communication functions that you might need

    result = np.empty_like(out)
    mp_comm.Allreduce(out, result, op=MPI.SUM)
    return result


def naive_collect_backward_output(
    output_grad: np.ndarray,
    mp_group_idx: int,
    mp_size: int,
):
    """The function for collecting layer fc2's output_grad across different nodes with naive model parallelism

    Parameters
    ----------
        output_grad : np.ndarray
            layer output_grad for a single node of shape (batch_size, out_dim)

        mp_group_idx : int
            The Model Parallel group idx

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_output_grad : np.ndarray
            collected layer output_grad across different nodes of shape (batch_size, part_out_dim)

    """

    

    # Hint: you might want to use np.split to get the collected_output_grad for each MP node

    return np.split(output_grad, mp_size, axis=1)[mp_group_idx]


def naive_collect_backward_x(
    grad_x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """The function for collecting layer fc2's grad_x across different nodes with naive model parallelism

    Parameters
    ----------
        grad_x : np.ndarray
            layer backward grad_x for a single node of shape (batch_size, in_dim)

        mp_comm : Communicator
            The Model Parallel communicator

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_grad_x : np.ndarray
            collected layer backward grad_x across different nodes of shape (batch_size, part_in_dim)

    """

    batch_size, in_dim = grad_x.shape
    part_in_dim = in_dim // mp_size

    parts = np.split(grad_x, mp_size, axis=1)          # mp_size x (batch, part_in_dim)
    send_buf = np.ascontiguousarray(
        np.concatenate([p.flatten() for p in parts])   # (batch*in_dim,)
    )

    recv_buf = np.empty(batch_size * part_in_dim, dtype=grad_x.dtype)
    mp_comm.Reduce_scatter(send_buf, recv_buf, op=MPI.SUM)

    return recv_buf.reshape(batch_size, part_in_dim)


def megatron_collect_backward_output(
    output_grad: np.ndarray,
    mp_group_idx: int,
    mp_size: int,
):
    """The function for collecting layer fc2's output_grad across different nodes with megatron-style model parallelism

    Parameters
    ----------
        output_grad : np.ndarray
            layer output_grad for a single node of shape (batch_size, out_dim)

        mp_group_idx : int
            The Model Parallel group idx

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_output_grad : np.ndarray
            collected layer output_grad across different nodes of shape (batch_size, part_out_dim)

    """

    

    # Hint: your implementation should be within one line of code

    return output_grad


def megatron_collect_backward_x(
    grad_x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """The function for collecting layer fc2's grad_x across different nodes with megatron-style model parallelism

    Parameters
    ----------
        grad_x : np.ndarray
            layer backward grad_x for a single node of shape (batch_size, in_dim)

        mp_comm : Communicator
            The Model Parallel communicator

        mp_size : int
            Model Parallel size

    Returns
    -------
        collected_grad_x : np.ndarray
            collected layer backward grad_x across different nodes of shape (batch_size, part_in_dim)

    """


    # Hint: your implementation should be within one line of code

    return grad_x


def collect_weight_grad(
    grad_w: np.ndarray,
    grad_b: np.ndarray,
    dp_comm,
):
    """The function for collecting weight gradients across data parallel nodes

    Parameters
    ----------
        grad_w : np.ndarray
            gradients value for fc weight on a single node of shape (in_dim, out_dim)

        grad_b : np.ndarray
            gradients value for fc bias on a single node of shape (1, out_dim)

        dp_comm : Communicator
            The Data Parallel communicator

    Returns
    -------
        collected_grad_w : np.ndarray
            collected gradients value of shape (in_dim, out_dim) for fc weight across different nodes

        collected_grad_b : np.ndarray
            collected gradients value of shape (1, out_dim) for fc bias across different nodes

    """

    collected_grad_w = np.empty_like(grad_w)
    collected_grad_b = np.empty_like(grad_b)

    dp_comm.Allreduce(grad_w, collected_grad_w, op=MPI.SUM)
    dp_comm.Allreduce(grad_b, collected_grad_b, op=MPI.SUM)

    dp_size = dp_comm.Get_size()
    # Average: each node saw 1/dp_size of the data, so average = correct full gradient
    return collected_grad_w / dp_size, collected_grad_b / dp_size
