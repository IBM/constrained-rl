# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import sys
import os
import numpy as np
import time

from ceres.tools import ExtraArgs
from baselines.common.cmd_util import make_mujoco_env
import baselines.common.tf_util as U
from ceres.envs import CeresEnv
from ceres.baselines.ceres.pposgd_ceres import build_policy_observation_filter

class DummyPolicy(object):
    '''
    A dummy policy that outputs either zero or random actions in the size expected by the environment
    '''

    def __init__(self, name, ob_space, ac_space):
        self.name = name
        self.ob_space = ob_space
        self.ac_space = ac_space

        self.ac_zero = np.zeros(self.ac_space.shape)
        self.vpred_zero = 0.

    def act(self, stochastic, ob):
        if stochastic:
            return self.ac_space.sample(), self.vpred_zero
        else:
            return self.ac_zero, self.vpred_zero

def main():
    '''
    Load and play trained policy
    '''
    log_root = os.path.join(os.getcwd(), 'logs')
    extra_args = ExtraArgs(log_root=log_root)

    env = make_mujoco_env(extra_args.env_id, extra_args.seed)

    if isinstance(env.unwrapped, CeresEnv) and (len(extra_args.trained_cnet) > 0):
        env.unwrapped.init_ceres()
        env.unwrapped.init_constraint_prediction(extra_args.trained_cnet)

    episode_lengths = np.zeros(extra_args.max_episodes)
    episode_rewards = np.zeros(extra_args.max_episodes)
    ob = env.reset()

    do_save_render = extra_args.render and len(extra_args.save_render) > 0
    if do_save_render:
        os.makedirs(extra_args.save_render, exist_ok=True)

    def save_render(i_step, max_step=300, verbose=True):
        n_digits = len(str(max_step))
        do_save_step = (max_step <= 0) or (i_step <= max_step)
        if do_save_render and do_save_step:
            path_save = os.path.join(extra_args.save_render, str(i_step).zfill(n_digits) + '.png')
            env.unwrapped.save_render(path_save, verbose=verbose)

    ob_space = env.unwrapped.observation_space
    ac_space = env.unwrapped.action_space
    ob_space, policy_observation_filter= build_policy_observation_filter(extra_args, ob_space)

    env.unwrapped.set_ineq_margin(extra_args.conservative_exploration)

    if len(extra_args.trained_policy) > 0:
        assert os.path.exists(extra_args.trained_policy), 'Invalid path to model: \'{0}\''.format(extra_args.trained_policy)
        from ceres.baselines.ceres.mlp_policy_saver import MlpPolicySaver
        from baselines.common import tf_util as U
        sess = U.single_threaded_session()
        sess.__enter__()

        def policy_fn(name, ob_space, ac_space):
            return MlpPolicySaver(name, ob_space=ob_space, ac_space=ac_space,
                hid_size=extra_args.policy_hidden_size, num_hid_layers=extra_args.policy_hidden_layers)
        pi = policy_fn('pi', ob_space, ac_space)

        U.initialize()
        pi.restore_model(extra_args.trained_policy, session=sess)
    else:
        print('Invalid model path \'{0}\', use dummy agent'.format(extra_args.trained_policy))
        pi = DummyPolicy('pi', ob_space, ac_space)

    time_total = 0.
    n_steps_global = -1
    for i_episode in range(extra_args.max_episodes):
        print('Episode {0}'.format(i_episode))
        time_episode_begin = time.time()
        ob = policy_observation_filter(ob)
        n_steps_global += 1
        if extra_args.render:
            env.render()
            save_render(n_steps_global)
        done = False
        ep_rew = 0.
        i_step = 0
        time.sleep(extra_args.play_step_duration)
        
        while not done:
            action, vpred = pi.act(True, ob)
            ob, rew, done, info = env.step(action)
            ob = policy_observation_filter(ob)
            ep_rew += rew
            i_step += 1
            n_steps_global += 1
            if extra_args.render:
                env.render()
                save_render(n_steps_global)
            time.sleep(extra_args.play_step_duration)
        episode_lengths[i_episode] = i_step
        episode_rewards[i_episode] = ep_rew
        time_episode = time.time() - time_episode_begin
        time_total += time_episode
        print('  Episode length: {0} (average {1:.1f}), episode reward {2:.1f} (average {5:.1f}), duration {3:.1f} ms (average {4:.1f})'.format(i_step, np.average(episode_lengths[:i_episode+1]), ep_rew, 1000.*time_episode, 1000.*time_total/(i_episode+1), np.average(episode_rewards[:i_episode+1])))
        ob = env.reset()


if __name__ == '__main__':
    main()
