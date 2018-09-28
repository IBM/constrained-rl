# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

from mpi4py import MPI
import numpy as np
from . import mpi_select

def mpi_mean_select(x, rank, root, destinations, n_processes,
                    axis=0, comm=None, keepdims=False):
    '''
    Compute a mean on a selection of processes instead of all
    '''
    x = np.asarray(x)
    assert x.ndim > 0
    if comm is None: comm = MPI.COMM_WORLD
    xsum = x.sum(axis=axis, keepdims=keepdims)
    n = xsum.size
    localsum = np.zeros(n+1, x.dtype)
    localsum[:n] = xsum.ravel()
    localsum[n] = x.shape[axis]
    #globalsum = np.zeros_like(localsum)
    #comm.Allreduce(localsum, globalsum, op=MPI.SUM)
    globalsum = mpi_select.Allreduce_select(comm, rank, root, destinations, localsum, tag_reduce=root, tag_bcast=root + n_processes)
    return globalsum[:n].reshape(xsum.shape) / globalsum[n], globalsum[n]

def mpi_moments_select(x, rank, root, destinations, n_processes,
                       axis=0, comm=None, keepdims=False):
    '''
    Compute a mean on a selection of processes instead of all
    '''
    x = np.asarray(x)
    assert x.ndim > 0
    mean, count = mpi_mean_select(x, rank, root, destinations, n_processes,
                                  axis=axis, comm=comm, keepdims=True)
    sqdiffs = np.square(x - mean)
    meansqdiff, count1 = mpi_mean_select(sqdiffs,
                                         rank, root, destinations, n_processes,
                                         axis=axis, comm=comm, keepdims=True)
    assert count1 == count
    std = np.sqrt(meansqdiff)
    if not keepdims:
        newshape = mean.shape[:axis] + mean.shape[axis+1:]
        mean = mean.reshape(newshape)
        std = std.reshape(newshape)
    return mean, std, count


