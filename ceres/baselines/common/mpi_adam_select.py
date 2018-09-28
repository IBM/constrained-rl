# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

### Control reduce / broadcast at the process level

from mpi4py import MPI
import baselines.common.tf_util as U
import tensorflow as tf
import numpy as np

from baselines.common.mpi_adam import MpiAdam
from . import mpi_select

class MpiAdamSelect(MpiAdam):
    '''
    Extend MpiAdam with parallelization across only a selection of processes (direct or recovery) instead of all
    '''

    def __init__(self, rank, root, group, var_list, *args, select_params=None, **kwargs):
        super().__init__(var_list, *args, **kwargs)
        self.init_select(rank, root, group)

    def init_select(self, rank, root, group):
        self.rank  = rank
        self.root  = root
        self.group = group
        self.destinations = [_e for _e in self.group if _e != self.root]
        self.n_processes = self.comm.Get_size()

    def update(self, localg, stepsize):
        if self.t % 100 == 0:
            self.check_synced()
        localg = localg.astype('float32')
        globalg = mpi_select.Allreduce_select(self.comm, self.rank, self.root, self.destinations, localg, tag_reduce=self.root, tag_bcast=self.root + self.n_processes)
        
        if self.scale_grad_by_procs:
            globalg /= len(self.group)

        self.t += 1
        a = stepsize * np.sqrt(1 - self.beta2**self.t)/(1 - self.beta1**self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = (- a) * self.m / (np.sqrt(self.v) + self.epsilon)
        self.setfromflat(self.getflat() + step)

    def sync(self):
        theta = self.getflat()
        theta = mpi_select.Bcast_select(self.comm, self.rank, self.root, self.destinations, theta, tag=self.root)
        self.setfromflat(theta)

    def check_synced(self):
        if self.rank == self.root: # this is root
            theta = self.getflat()
            theta = mpi_select.Bcast_select(self.comm, self.rank, self.root, self.destinations, theta, tag=self.root)
        else:
            thetalocal = self.getflat()
            thetaroot = np.empty_like(thetalocal)
            thetaroot = mpi_select.Bcast_select(self.comm, self.rank, self.root, self.destinations, thetaroot, tag=self.root)
            assert (thetaroot == thetalocal).all(), (thetaroot, thetalocal)
