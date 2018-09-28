# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import os
from mpi4py import MPI
from baselines.common.mpi_fork import mpi_fork
from baselines.common import tf_util as U
from baselines import logger
from . import pposgd_ceres

from baselines.common.cmd_util import make_mujoco_env
from .mlp_policy_saver import MlpPolicySaver
from ceres.envs import CeresEnv
from ceres import ConstraintNetworkMLP, ConstraintConfig
from ceres import ConstraintDemonstrationBuffer

def build_log_dirs(path_xp, rank, is_direct_policy):
    worker_name = 'worker_{0}_{1}'.format(rank, 'direct' if is_direct_policy else 'recovery')
    worker_dir = os.path.join(path_xp, worker_name)
    worker_policy_dir = os.path.join(worker_dir, 'policy')
    worker_constraints_dir = os.path.join(worker_dir, 'constraints')
    return worker_name, worker_dir, worker_policy_dir, worker_constraints_dir

def main():
    '''
    Initialize CERES environment and launch policy and constraint learning
    or restart from a previous training session.
    '''
    from ceres.tools import ExtraArgs
    log_root = os.path.join(os.getcwd(), 'logs')
    extra_args = ExtraArgs(log_root=log_root)

    n_agents_total = extra_args.n_direct + extra_args.n_recovery
    whoami  = mpi_fork(n_agents_total)
    if whoami == "parent":
        return
    sess = U.single_threaded_session()
    sess.__enter__()

    # Synchronize log directory between agents
    rank = MPI.COMM_WORLD.Get_rank()
    is_direct_policy = rank < extra_args.n_direct
    if rank == 0:
        path_xp = extra_args.path_xp
        for dest_rank in range(n_agents_total):
            send_buffer = [path_xp]
            MPI.COMM_WORLD.send(send_buffer, dest=dest_rank, tag=rank)
    else:
        recv_buffer = MPI.COMM_WORLD.recv(source=0, tag=0)
        path_xp = recv_buffer[0]

    # Find root processes for direct and recovery
    if is_direct_policy:
        root_rank = 0
    else:
        root_rank = extra_args.n_direct

    worker_name, worker_dir, worker_policy_dir, worker_constraints_dir = build_log_dirs(path_xp, rank, is_direct_policy)
    logger.configure(dir=worker_dir)

    if not rank == 0: # only log first direct
        logger.set_level(logger.DISABLED)

    workerseed = extra_args.seed + 10000 * rank
    assert len(extra_args.env_id) > 0, 'Missing argument --env_id'
    env = make_mujoco_env(extra_args.env_id, workerseed)
    assert isinstance(env.unwrapped, CeresEnv), 'Env {0} should be an instance of CeresEnv'.format(type(env))
    env.unwrapped.init_ceres(is_recovery_mode=(not is_direct_policy))

    # Setup restoration parameters from previous logs
    if len(extra_args.continue_ceres_training) > 0:
        assert os.path.isdir(extra_args.continue_ceres_training), 'Could not find log directory: {0}'.format(extra_args.continue_ceres_training)
        # All direct share one policy, all recovery share another
        _, _, extra_args.trained_policy, _ = build_log_dirs(extra_args.continue_ceres_training, root_rank, is_direct_policy)
        # All agents share a single constraint network
        _, _, _, extra_args.trained_cnet = build_log_dirs(extra_args.continue_ceres_training, 0, True)
        # All agents have separate demonstration buffers
        _, _, _, extra_args.constraint_demonstration_buffer = build_log_dirs(extra_args.continue_ceres_training, rank, is_direct_policy)

    def policy_fn(name, ob_space, ac_space):
        policy = MlpPolicySaver(name, ob_space=ob_space, ac_space=ac_space,
                                hid_size=extra_args.policy_hidden_size, num_hid_layers=extra_args.policy_hidden_layers)
        policy.init_saver(worker_policy_dir, session=sess, max_to_keep=extra_args.backup_keep)
        return policy

    # Initialize backup directories
    os.makedirs(worker_constraints_dir, exist_ok=True)
    if len(extra_args.trained_cnet) > 0:
        cnet_config = ConstraintConfig.from_backup(extra_args.trained_cnet)
    else:
        cnet_config = ConstraintConfig.from_extra_args(extra_args)
    if rank == 0:
        cnet_config.save(worker_constraints_dir)
    cnet = ConstraintNetworkMLP(env.observation_space, env.action_space, cnet_config)
    cnet.init_saver(worker_constraints_dir, session=sess, max_to_keep=extra_args.backup_keep)
    env.unwrapped.init_constraint_prediction(cnet, session=sess)

    constraint_demonstration_buffer = ConstraintDemonstrationBuffer(extra_args.constraint_demonstration_buffer_size)
    constraint_demonstration_buffer.init_saver(worker_constraints_dir)

    # Check end criterion
    possible_end_criteria = ['max_iterations', 'max_timesteps', 'max_episodes', 'max_seconds']
    active_end_criteria = [_k for _k in possible_end_criteria if getattr(extra_args, _k) > 0]
    n_end_criteria = len(active_end_criteria)
    if extra_args.max_iterations == 0:
        raise ValueError('Specify one end criterion out of {0}'.format(possible_end_criteria))
    else:
        assert n_end_criteria == 1, 'Only one time constraint permitted but {0} specified: {1}'.format(n_end_criteria, active_end_criteria)

    # Start training!
    pposgd_ceres.learn(env, policy_fn,
            max_timesteps=extra_args.max_timesteps,
            max_iters=extra_args.max_iterations,
            max_episodes=extra_args.max_episodes,
            max_seconds=extra_args.max_seconds,
            timesteps_per_actorbatch=extra_args.timesteps_per_actorbatch,
            clip_param=0.2, entcoeff=extra_args.policy_entcoeff,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule=extra_args.policy_learning_rate_schedule,
            extra_args=extra_args, cnet=cnet, constraint_demonstration_buffer=constraint_demonstration_buffer,
        )
    env.close()

    if rank == 0:
        print('Done! Logs are located in {0}'.format(path_xp))

if __name__ == '__main__':
    main()
