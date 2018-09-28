# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import numpy as np

'''
These reproduce the behavior of MPI functions, except applied to a subset of available processes.
In addition, all functions return something instead of acting on argument buffers directly.
'''

def Bcast_select(comm, rank, root, destinations, data_buffer, tag=0):
    if rank == root:
        for dest_rank in destinations:
            comm.send(data_buffer, dest=dest_rank, tag=tag)
        return data_buffer
    else:
        recv_buffer = comm.recv(source=root, tag=tag)
        assert len(recv_buffer) == len(data_buffer)
        return recv_buffer

def Reduce_select(comm, rank, root, destinations, val_buffer, sum_buffer=None, tag=0):
    if sum_buffer is None:
        sum_buffer = np.zeros_like(val_buffer)
    if rank == root:
        sum_buffer += val_buffer
        for dest_rank in destinations:
            recv_buffer = comm.recv(source=dest_rank, tag=root)
            sum_buffer += recv_buffer
    else:
        comm.send(val_buffer, dest=root, tag=root)
    return sum_buffer

def Allreduce_select(comm, rank, root, destinations, val_buffer, tag_reduce=0, tag_bcast=0):
    assert tag_reduce != tag_bcast
    sum_buffer = Reduce_select(comm, rank, root, destinations, val_buffer, tag=tag_reduce)
    sum_buffer = Bcast_select(comm, rank, root, destinations, sum_buffer, tag=tag_bcast)
    return sum_buffer

def allgather_select(comm, rank, root, destinations, data_buffer, tag=0):
    # Gather everything to root
    index_map = {_v: _i for _i, _v in enumerate([root] + list(destinations))}
    gather_buffer = [None] * len(index_map)
    if rank == root:
        gather_buffer[index_map[root]] = data_buffer
        for dest_rank in destinations:
            recv_buffer = comm.recv(source=dest_rank, tag=tag)
            gather_buffer[index_map[dest_rank]] = recv_buffer
    else:
        comm.send(data_buffer, dest=root, tag=tag)
    # Broadcast to destinations
    allgather_buffer = Bcast_select(comm, rank, root, destinations, gather_buffer, tag=tag)
    return allgather_buffer

