# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

from baselines import logger
import baselines.common.tf_util as U
import numpy as np
from ceres.baselines.common import mpi_select
from ceres.baselines.common.mpi_adam_select import MpiAdamSelect
from ceres.baselines.common.mpi_moments_select import mpi_moments_select
from ceres import ConstraintDemonstration, ConstraintDemonstrationTrajectory, ConstraintDemonstrationBuffer
from .ceres_logic import CeresLogic
from .constraint_trainer import ConstraintTrainer

from ceres.baselines.ppo1.pposgd_simple_helper import build_policy_training_vars, build_counters, adjust_policy_learning_rate, update_policy, log_iter_info, calc_end_training
from .pposgd_ceres_helper import update_constraint_activation_probability, build_policy_observation_filter, build_mpi_vars, save_models_and_data

def traj_segment_generator(pi, env, horizon,
                           ceres_logic, is_direct_policy, policy_observation_filter,
                           stochastic=True, render=False):
    '''
    Sample trajectories and collect positive/negative/uncertain demonstrations for constraint learning
    '''
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()
    policy_ob = policy_observation_filter(ob)
    if render:
        env.render()
    i_iteration = 0

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([policy_ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    demonstration_trajectory = [] # store sampled demonstrations here
    uncertain_demonstration_trajectories = [] # demonstrations that cannot yet be identified as positive or negative will go here
    snapshot = env.unwrapped.calc_snapshot()
    recovery_info = env.unwrapped.recovery_info

    while True:
        prevac = ac
        policy_ob = policy_observation_filter(ob)
        ac, vpred = pi.act(stochastic, policy_ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            res = {'ob' : obs, 'rew' : rews, 'vpred' : vpreds, 'new' : news,
                    'ac' : acs, 'prevac' : prevacs, 'nextvpred': vpred * (1 - new),
                    'ep_rets' : ep_rets, 'ep_lens' : ep_lens,
            }
            res['uncertain_demonstration_trajectories'] = uncertain_demonstration_trajectories
            yield res
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            uncertain_demonstration_trajectories = []
            i_iteration += 1
        i = t % horizon
        obs[i] = policy_ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob_new, rew, new, info = env.step(ac)
        if render:
            env.render()
        rews[i] = rew
        ac_constrained = info[env.unwrapped.info_key_constrained_action]

        ceres_demonstration = ConstraintDemonstration(state=ob, snapshot=snapshot, action=ac_constrained)
        demonstration_trajectory.append(ceres_demonstration)
        ob = ob_new
        snapshot = env.unwrapped.calc_snapshot()

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            # Add final demonstration as terminal (state without action)
            demonstration_trajectory.append(ConstraintDemonstration(state=ob, snapshot=snapshot, is_terminal=True))
            # Sort demonstrations into positive and negative
            ceres_logic.process_trajectory(ConstraintDemonstrationTrajectory(demonstration_trajectory),
                                           info[env.unwrapped.info_key_failure], info[env.unwrapped.info_key_success],
                                           uncertain_demonstration_trajectories,
                                           env.unwrapped.recovery_info, is_direct_policy,
                                           remove_reference_trajectory_if_emptied=True,
                                           increment_reset_count_on_change=i_iteration)
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            # Reset trajectory parameters
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
            snapshot = env.unwrapped.calc_snapshot()
            if render:
                env.render()
            demonstration_trajectory = []
        t += 1

def learn(env, policy_fn, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        extra_args, cnet, constraint_demonstration_buffer,
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant' # annealing for stepsize parameters (epsilon and adam)
):

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, 'Only one time constraint permitted'

    # Train different networks across processes
    mpi_comm, mpi_rank, is_direct_policy, mpi_root, mpi_group, mpi_destinations, mpi_n_processes, is_root, cnet_recovery_id_in_direct_exchange_ids, cnet_exchange_ids, n_exchange_processes = build_mpi_vars(extra_args)

    # Setup observation filtering (use partial state information)
    policy_ob_space = env.observation_space
    policy_ac_space = env.action_space
    policy_ob_space, policy_observation_filter = build_policy_observation_filter(extra_args, policy_ob_space)
    pi = policy_fn('pi', policy_ob_space, policy_ac_space) # Construct network for new policy
    oldpi = policy_fn('oldpi', policy_ob_space, policy_ac_space) # Network for old policy

    # Create policy optimizers
    policy_loss_names, policy_var_list, policy_lossandgrad, policy_adam, policy_assign_old_eq_new, policy_compute_losses = build_policy_training_vars(pi, oldpi, clip_param, entcoeff, adam_epsilon)
    # Use rank-selective Adam to train direct and recovery with separate data
    policy_adam = MpiAdamSelect(mpi_rank, mpi_root, mpi_group, policy_var_list, epsilon=adam_epsilon)
    mpi_moments_fn = lambda losses: mpi_moments_select(losses, mpi_rank, mpi_root, mpi_destinations, mpi_n_processes, axis=0)
    allgather_fn = lambda x: mpi_select.allgather_select(mpi_comm, mpi_rank, mpi_root, mpi_destinations, x, tag=mpi_root)

    # Constraints
    cnet = env.unwrapped.cnet
    last_backup_time = None

    constraint_trainer = ConstraintTrainer(extra_args, logger,
                                           cnet, constraint_demonstration_buffer,
                                           mpi_comm, mpi_rank, is_direct_policy,
                                           cnet_recovery_id_in_direct_exchange_ids,
                                           cnet_exchange_ids, n_exchange_processes,
                                           adam_epsilon=adam_epsilon)

    # Enable conservative exploration (force margin w.r.t. constraints)
    env.unwrapped.set_ineq_margin(extra_args.conservative_exploration)
    # Set initial constraint activation probability to zero
    update_constraint_activation_probability(env, extra_args, logger, is_direct_policy, True, 0., 0.)

    U.initialize()
    if len(extra_args.trained_policy) > 0:
        pi.restore_model(extra_args.trained_policy)
        oldpi.restore_model(extra_args.trained_policy, backup_network_id='pi')
    if len(extra_args.trained_cnet) > 0:
        cnet.restore_model(extra_args.trained_cnet)
    if len(extra_args.constraint_demonstration_buffer) > 0:
        constraint_demonstration_buffer.restore_buffer(extra_args.constraint_demonstration_buffer, keep_size=False, verbose=True)

    policy_adam.sync()
    constraint_trainer.init()

    # Prepare for rollouts
    ceres_logic = CeresLogic(env, constraint_demonstration_buffer, extra_args)
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch,
                                     ceres_logic,
                                     is_direct_policy,
                                     policy_observation_filter,
                                     stochastic=True,
                                     render=(extra_args.render and (mpi_rank == 0)))

    iters_so_far, episodes_so_far, timesteps_so_far, tstart, lenbuffer, rewbuffer = build_counters()

    do_train_policy  = True
    do_sync_recovery = True
    do_train_cnet    = True

    if extra_args.n_recovery == 0:
        do_sync_recovery = False
        do_train_cnet    = False # turn it back on if only train constraints

    if extra_args.only_train_constraints:
        do_train_policy  = False
        do_sync_recovery = False
        do_train_cnet    = True
        assert len(extra_args.constraint_demonstration_buffer) > 0, 'Required constraint demonstration buffer'
        max_iters, max_timesteps, max_episodes, max_seconds = 1, 0, 0, 0

    if extra_args.only_train_policy:
        do_train_policy  = True
        do_sync_recovery = False
        do_train_cnet    = False

    while True:
        if callback: callback(locals(), globals())

        logger.log('********** Begin iteration {0} ************'.format(iters_so_far))

        n_reference_trajectories_before_sampling = len(env.unwrapped.reference_trajectories)

        if do_train_policy:
            # Collect new trajectories and update policy
            seg = seg_gen.__next__()
            policy_cur_lrmult = adjust_policy_learning_rate(schedule, max_timesteps, timesteps_so_far, max_episodes, episodes_so_far, max_iters, iters_so_far)
            vpredbefore, tdlamret, optim_batchsize = update_policy(pi, seg, gamma, lam,
                                                                   logger, optim_epochs, optim_batchsize, optim_stepsize, policy_cur_lrmult,
                                                                   policy_loss_names, policy_lossandgrad, policy_adam, policy_assign_old_eq_new, policy_compute_losses,
                                                                   mpi_moments_fn)

        if do_train_policy and do_sync_recovery:
            # Transfer uncertain demonstrations from direct to recovery agents
            constraint_trainer.synchronize_recovery_trajectories(env, seg, n_reference_trajectories_before_sampling)

        # Compute constraint losses on the newly collected data, prior to training
        do_train_cnet, activation_probability_before = constraint_trainer.prepare_constraint_update(do_train_cnet, iters_so_far)

        # Train constraints on the new data and return final losses
        do_train_cnet, activation_probability_after = constraint_trainer.update_constraint_network(do_train_cnet)

        # Change the environment constraint activation probability dependending on the constraint prediction accuracy
        update_constraint_activation_probability(env, extra_args, logger, is_direct_policy, do_train_cnet,
                                                 activation_probability_before, activation_probability_after)

        # Log iteration results
        if do_train_policy:
            episodes_so_far, timesteps_so_far = log_iter_info(lenbuffer, rewbuffer, tstart,
                                                              vpredbefore, tdlamret, seg,
                                                              episodes_so_far, timesteps_so_far,
                                                              is_root, allgather_fn)
        iters_so_far += 1
        end_training = calc_end_training(max_timesteps, timesteps_so_far,
                                         max_episodes, episodes_so_far,
                                         max_iters, iters_so_far,
                                         max_seconds, tstart)

        # Save models and data
        last_backup_time = save_models_and_data(extra_args, iters_so_far, end_training, last_backup_time,
                                                is_root, mpi_rank, pi, cnet, constraint_demonstration_buffer)

        if end_training:
            break
