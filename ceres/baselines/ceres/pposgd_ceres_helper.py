# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

from baselines import logger
import baselines.common.tf_util as U
import numpy as np
import time
from mpi4py import MPI
import gym

def update_constraint_activation_probability(env, extra_args, logger, is_direct_policy, do_train_cnet,
                                             activation_probability_before, activation_probability_after):
    '''
    Update environment constraint activation probability using constraint accuracy before or after training
    '''
    activation_probability = extra_args.constant_constraint_activation
    if len(extra_args.adaptive_constraint_activation) > 0:
        do_use_prior_accuracy_as_activation_probability = 'prior' in extra_args.adaptive_constraint_activation
        if do_use_prior_accuracy_as_activation_probability or (not do_train_cnet):
            activation_probability = activation_probability_before
        else:
            activation_probability = activation_probability_after
    if (not is_direct_policy) and extra_args.unconstrained_recovery:
        activation_probability = 0.
    if activation_probability is not None:
        logger.log('Set constraint activation probability to {0:.1f} %'.format(activation_probability * 100.))
        env.unwrapped.set_constraint_activation_probability(activation_probability)

def check_time_between_backups(extra_args, last_backup_time=None):
    '''
    Only write backups every min_time_between_backups
    '''
    time_now = time.time()
    if last_backup_time is not None:
        time_since_last = time_now - last_backup_time
        do_save_backup = time_since_last > extra_args.min_time_between_backups
    else:
        do_save_backup = True
    if do_save_backup:
        last_backup_time = time_now
    return do_save_backup, last_backup_time

def build_policy_observation_filter(extra_args, ob_space):
    '''
    If extra_args.policy_observation_filter is a string of the form "1:3:6", only provide the policy with observations number 1, 3 and 6
    '''
    if len(extra_args.policy_observation_filter) == 0:
        observation_filter = lambda ob: ob
        ob_space_filtered = ob_space
    else:
        indices = [int(_v) for _v in extra_args.policy_observation_filter.split(':')]
        observation_filter = lambda ob: np.array([ob[_i] for _i in indices], dtype=ob.dtype)
        low_filtered = observation_filter(ob_space.low)
        high_filtered = observation_filter(ob_space.high)
        ob_space_filtered = gym.spaces.Box(low=low_filtered, high=high_filtered, dtype=ob_space.dtype)
    return ob_space_filtered, observation_filter

def build_mpi_vars(extra_args):
    '''
    Initialize process indices across direct and recovery agents
    '''
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    is_direct_policy = mpi_rank < extra_args.n_direct

    mpi_root_direct = 0
    mpi_group_direct = list(range(extra_args.n_direct))
    mpi_root_recovery = extra_args.n_direct
    mpi_group_recovery = list(range(extra_args.n_direct, extra_args.n_direct + extra_args.n_recovery))
    if is_direct_policy:
        mpi_root = mpi_root_direct
        mpi_group = mpi_group_direct
    else:
        mpi_root = mpi_root_recovery
        mpi_group = mpi_group_recovery
    mpi_destinations = [_e for _e in mpi_group if _e != mpi_root]
    mpi_n_processes = extra_args.n_direct + extra_args.n_recovery
    is_root = mpi_rank == mpi_root

    if extra_args.n_recovery > 0:
        # Correspondences between direct and recovery agents for CNet data exchange
        cnet_exchange_ids = {_i: [] for _i in mpi_group_direct + mpi_group_recovery}
        for _i in range(max(len(mpi_group_direct), len(mpi_group_recovery))):
            _i_direct = mpi_group_direct[_i % len(mpi_group_direct)]
            _i_recovery = mpi_group_recovery[_i % len(mpi_group_recovery)]
            if not (_i_recovery in cnet_exchange_ids[_i_direct]):
                cnet_exchange_ids[_i_direct].append(_i_recovery)
            if not (_i_direct in cnet_exchange_ids[_i_recovery]):
                cnet_exchange_ids[_i_recovery].append(_i_direct)

        # Also get the index of each recovery process within those associated to the corresponding direct process (re-read this several times)
        cnet_recovery_id_in_direct_exchange_ids = {_i: {} for _i in mpi_group_recovery}
        for _i_recovery in mpi_group_recovery:
            for _i_direct in cnet_exchange_ids[_i_recovery]:
                cnet_recovery_id_in_direct_exchange_ids[_i_recovery][_i_direct] = cnet_exchange_ids[_i_direct].index(_i_recovery)
        n_exchange_processes = len(cnet_exchange_ids[mpi_rank])
    else:
        cnet_exchange_ids = None
        cnet_recovery_id_in_direct_exchange_ids = None
        n_exchange_processes = None

    return mpi_comm, mpi_rank, is_direct_policy, mpi_root, mpi_group, mpi_destinations, mpi_n_processes, is_root, cnet_recovery_id_in_direct_exchange_ids, cnet_exchange_ids, n_exchange_processes

def save_models_and_data(extra_args, iters_so_far, end_training, last_backup_time,
                         is_root, mpi_rank, pi, cnet, constraint_demonstration_buffer):
    '''
    Save policy network, constraint network and constraint demonstration buffer
    '''
    do_save_at_all = extra_args.backup_frequency > 0
    do_save_this_iter = (((iters_so_far - 1) % extra_args.backup_frequency) == 0) or end_training
    do_save_this_time, last_backup_time = check_time_between_backups(extra_args, last_backup_time)
    do_save_policy      = not extra_args.only_train_constraints
    do_save_constraints = not extra_args.only_train_policy
    do_save_buffer      = not (extra_args.only_train_policy or extra_args.only_train_constraints)
    if do_save_at_all and do_save_this_iter and do_save_this_time:
        if do_save_policy and is_root:
            # save direct and recovery policies separatery
            pi.save_model(global_step=(iters_so_far-1), verbose=True)
        if do_save_constraints and (mpi_rank == 0):
            # same CNet for all agents
            cnet.save_model(global_step=(iters_so_far-1), verbose=True)
        if do_save_buffer:
            # different buffers for all agents
            constraint_demonstration_buffer.write(verbose=is_root)
    return last_backup_time
