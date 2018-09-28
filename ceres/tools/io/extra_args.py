# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import os
import datetime
import argparse
import shutil
import numpy as np

class ExtraArgs(object):
    '''
    A simple class to parse and check command-line arguments for CERES
    '''

    def __init__(self, log_root='/tmp', args=None,
                 **kwargs):
        if args is not None:
            self.args = args
        else:
            self.args = self.parse_args()
            for _k, _v in kwargs.items():
                assert hasattr(self.args, _k)
                setattr(self.args, _k, _v)
            self.check_args()
        self.import_env_module(module_id=self.args.module_id)
        self.timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.log_root = log_root
        self.path_xp = self.build_path()

    def __getattr__(self, _k):
        return getattr(self.args, _k)

    def build_path(self):
        if len(self.args.output) > 0:
            xp_dirname = self.args.output
        else:
            xp_dirname = self.timestamp
        path_xp = os.path.join(self.log_root, xp_dirname)
        if os.path.exists(path_xp):
            exists_str = 'Log path already exists: {0}'.format(path_xp)
            if self.args.overwrite:
                path_move = path_xp + '_moved_{0}'.format(self.timestamp)
                exists_str += '\n  Moved existing logs to: {0}'.format(path_move)
            else:
                exists_str += '\n  Remove dir manually or run with --overwrite'
                raise ValueError(exists_str)
            shutil.move(path_xp, path_move)
            print(exists_str)
        return path_xp

    @staticmethod
    def parse_env_module(env_id):
        if ':' in env_id:
            module_id, env_id = env_id.split(':')
        else:
            module_id = ''
        return env_id, module_id

    def check_args(self):
        self.args.env_id, self.args.module_id = self.parse_env_module(self.args.env_id)
        self.args.cnet_hidden_layers = list(map(int, self.args.cnet_hidden_layers.split(',')))

        self.args.cnet_loss_weights = {}
        for loss_weight_arg_str in self.args.cnet_loss:
            try:
                loss_name, loss_weight = loss_weight_arg_str.split(':')
                loss_weight = float(loss_weight)
            except:
                raise ValueError('Invalid --loss_weight argument {0}, excepted format <loss_name>:<loss_weight>'.format(loss_weight_arg_str))
            self.args.cnet_loss_weights[loss_name] = loss_weight
        assert not (self.args.cnet_spherical_coordinates and self.args.cnet_normalize_ineq_mat), 'Cannot have simultaneously --cnet_spherical_coordinates and --cnet_normalize_ineq_mat'
        assert self.args.n_direct > 0, 'Set at least one direct agent'
        if self.args.n_recovery is None: # By default, set equal number of direct and recovery agents
            self.args.n_recovery = self.args.n_direct

        assert (self.args.constant_constraint_activation is None) or (len(self.args.adaptive_constraint_activation) == 0), 'Cannot set both constant and adaptive constraint activation probability'
        # Plot
        if (self.args.plot_average % 2) == 0: self.args.plot_average += 1 # make it odd

    @staticmethod
    def import_env_module(env_id=None, module_id=None):
        if module_id is None:
            assert env_id is not None
            env_id, module_id = ExtraArgs.parse_env_module(env_id)
        if len(module_id) > 0:
            import importlib
            print('Import module {0}'.format(module_id))
            importlib.import_module(module_id)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        args, unprocessed_args = parser.parse_known_args()

        # Base reinforcement learning parameters
        parser.add_argument('-e', '--env_id', default='', help='Environment name')
        parser.add_argument('--timesteps_per_actorbatch', default=1024, type=int, help='timesteps per actor per training batch')
        parser.add_argument('--max_iterations', default=0, type=int, help='maximum total number of training iterations')
        parser.add_argument('--max_episodes', default=0, type=int, help='maximum total number of episodes')
        parser.add_argument('--max_timesteps', default=0, type=int, help='maximum total number of timesteps')
        parser.add_argument('--max_seconds', default=0, type=int, help='maximum training time in seconds')
        parser.add_argument('--seed', default=0, type=int, help='random seed')
        parser.add_argument('--backup_frequency', default=1, type=int, help='save every n iterations')
        parser.add_argument('--backup_keep', default=1, type=int, help='number of backups to keep, set to 0 to keep all')
        parser.add_argument('--min_time_between_backups', default=60., type=float, help='minimum time in seconds between model backups')
        parser.add_argument('--continue_ceres_training', default='', help='root directory for CERES logs')
        parser.add_argument('--output', default='', help='output log dir')
        parser.add_argument('--render', action='store_true', help='render')
        parser.add_argument('--save_render', default='', help='directory to save render')
        parser.add_argument('--policy_hidden_size', type=int, default=64)
        parser.add_argument('--policy_hidden_layers', type=int, default=2)
        parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0.)
        parser.add_argument('--policy_learning_rate_schedule', default='linear', choices=['constant', 'linear'], help='policy learning rate schedule',)

        # Constraint network architecture
        parser.add_argument('--cnet_n_ineq', default=2, type=int, help='Number of inequality constraints')
        parser.add_argument('--cnet_batch_size', default=64, type=int, help='Batch size')
        parser.add_argument('--cnet_hidden_layers', default='64,64', help='Number of inequality constraints')
        parser.add_argument('--cnet_spherical_coordinates', action='store_true', help='Inequality matrix is first predicted as (n-1)-dim spherical coordinates')
        parser.add_argument('--cnet_normalize_ineq_mat', action='store_true', help='Normalize each row of the inequality matrices')
        parser.add_argument('--cnet_predict_interior_point', action='store_true', help='Predict one point satisfying all constraints')
        parser.add_argument('--cnet_interior_point_margin_min', default=0.1, type=float, help='minimum distance to interior point')
        parser.add_argument('--cnet_interior_point_margin_max', default=1., type=float, help='maximum distance to interior point')
        parser.add_argument('--cnet_interior_point_max', default=1., type=float, help='maximum value for the interior point per action component')
        parser.add_argument('--cnet_loss', default=[], action='append', help='loss weights')

        # CERES
        parser.add_argument('-n', '--n_direct', default=1, type=int, help='number of agents for direct reinforcement learning')
        parser.add_argument('--n_recovery', default=None, type=int, help='number of agents learning recovery')
        parser.add_argument('--constant_constraint_activation', default=None, type=float, help='constant constraint activation probability')
        parser.add_argument('--adaptive_constraint_activation', type=str, choices=['average', 'positive', 'negative', 'prior_average', 'prior_positive', 'prior_negative', 'prior_min'], default='', help='which constraint accuracy to use, if not empty')
        parser.add_argument('--interrupt_constraint_training', default='', help='condition for stopping CNet training')
        parser.add_argument('--policy_observation_filter', default='', help='use only these state elements')
        parser.add_argument('--only_train_constraints', action='store_true', help='only train constraint network in CERES')
        parser.add_argument('--only_train_policy', action='store_true', help='only train policy in CERES')
        parser.add_argument('--constraint_demonstration_buffer', default='', help='path to constraint demonstration buffer to restore')
        parser.add_argument('--constraint_demonstration_buffer_size', default=2048, type=int, help='Experience constraint demonstration size')
        parser.add_argument('--cnet_decay_epochs', default=0, type=int, help='decay CNet learning rate every N epochs without improvement')
        parser.add_argument('--cnet_decay_max', default=0.01, type=float, help='Keep learning rate >= this')
        parser.add_argument('--early_stop_positive', default=1.0, type=float)
        parser.add_argument('--early_stop_negative', default=1.0, type=float)
        parser.add_argument('--conservative_exploration', default=0.09, type=float, help='remove this from ineq vec')
        parser.add_argument('--max_recovery_attempts', default=10, type=int, help='number of recovery attempts per reference trajectory')
        parser.add_argument('--unconstrained_recovery', action='store_true', help='do not constrain recovery agent')
        parser.add_argument('--cnet_training_epochs', default=10, type=int, help='Number of training epoch')
        parser.add_argument('--cnet_training_batches', default=0, type=int, help='If > 0, maximum number of batches per epoch')
        parser.add_argument('--cnet_learning_rate', default=1.e-3, type=float)
        parser.add_argument('--cnet_improvement_metric', type=str, default='total_loss', choices=['mean_accuracy', 'min_accuracy', 'total_loss', 'mean_loss', 'max_loss'], help='Improvement metric for LR annealing')

        # Write, restore and replay
        parser.add_argument('--play_step_duration', default=0., type=float, help='wait duration in seconds when replaying baselines')
        parser.add_argument('--trained_policy', default='', help='load policy model backup')
        parser.add_argument('--trained_cnet', default='', help='Path to constraint network configuration')
        parser.add_argument('--overwrite', action='store_true', help='automatically moves log dir if it already exists')

        # Plot
        parser.add_argument('--plot_average', default=401, type=int, help='Moving average over N episodes')
        parser.add_argument('--plot_path', default=[], action='append', help='Path to logs')

        args = parser.parse_args(unprocessed_args)
        return args
