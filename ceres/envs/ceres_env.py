# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

from .resetter import ResetterEnvCeres
from .constrained import ConstrainedEnvNetwork
import numpy as np
import gym

class CeresEnv(ResetterEnvCeres, ConstrainedEnvNetwork):
    '''
    Base class for CERES-compatible environments, with support for snapshotting and constraint prediction
    '''

    # ResetterEnv parameters
    max_reference_trajectories = 1024 # -1 to load all available episodes

    ### Solver parameters
    return_zero_if_opt_fails = False
    has_ineq = True
    has_eq   = False

    # Recovery parameters
    recovery_reward_alive = 1.
    info_key_constrained_action = 'constrained_action'

    is_ceres_initialized = False # by default, use base environment (no special reset or constraints)

    def init_ceres(self, is_recovery_mode=False):
        '''
        Setup CERES-specific behavior.
        If this function is not called, function as the base environment, without constraints or recovery.
        '''
        if not self.is_ceres_initialized: # avoid double initialization through main class super
            self.check_base_attributes()
            self.init_overloading()
            self.init_recovery()
            ResetterEnvCeres.__init__(self)
            ConstrainedEnvNetwork.__init__(self)
            self.is_ceres_initialized = True
            self.is_recovery_mode = is_recovery_mode
            assert self.max_reference_trajectories > 0
            self.enable_constraints = True
            self.set_constraint_activation_probability(1.)

    def set_constraint_activation_probability(self, val):
        self.constraint_activation_probability = val

    def check_base_attributes(self):
        # Old gym environments use _reset instead of reset directly
        if hasattr(self, 'reset'):
            self.reset_function_name = 'reset'
        else:
            assert hasattr(self, '_reset'), 'Could find neither \'reset\' nor \'_reset\' base function.'
            self.reset_function_name = '_reset'
        # Old gym environments use _step instead of step directly
        if hasattr(self, 'step'):
            self.step_function_name = 'step'
        else:
            assert hasattr(self, '_step'), 'Could find neither \'step\' nor \'_step\' base function.'
            self.step_function_name = '_step'
        # The base environment also needs to be able to calculate snapshots and reset to given snapshots
        assert hasattr(self, 'calc_snapshot'), 'Could not find base calc_snapshot function'
        assert hasattr(self, 'reset_and_restore'), 'Could not find base reset_and_restore function'

    def init_overloading(self):
        '''
        Replace base environment reset and step with CERES-specific functions
        '''
        self.init_overloading_reset()
        self.init_overloading_step()

    def init_overloading_reset(self):
        self.reset_base = getattr(self, self.reset_function_name)
        setattr(self, self.reset_function_name, self.reset_ceres)

    def init_overloading_step(self):
        self.step_base = getattr(self, self.step_function_name)
        setattr(self, self.step_function_name, self.step_ceres)

    def step_ceres(self, action_raw):
        '''
        Depending on the constraint activation probability, correct the input action, play the corrected action and update constraints.
        For recovery, change the reward and end condition.
        '''
        do_enable_constraints_this_step = (self.constraint_activation_probability == 1.) or (np.random.rand() < self.constraint_activation_probability)
        if self.enable_constraints and do_enable_constraints_this_step:
            action_constrained, success, viol = self.correct_action(action_raw)
        else:
            action_constrained = action_raw
        state, reward, done, info = self.step_base(action_constrained)
        self.update_ineq_matrices(state)
        info[self.info_key_constrained_action] = action_constrained
        if self.is_recovery_mode:
            self.n_recovery_steps += 1
            is_max_recovery_steps = self.n_recovery_steps == self.max_recovery_steps
            if info[self.info_key_failure]:
                reward = self.recovery_reward_failure
            else:
                reward = self.recovery_reward_alive
            done = done or is_max_recovery_steps
        return state, reward, done, info

    def reset_ceres(self):
        '''
        Restore a reference snapshot when available (e.g., in recovery mode), otherwise use the base environment reset, and predict new constraints
        '''
        if len(self.reference_trajectories) == 0:
            state = self.reset_base()
            self.recovery_info = None
        else:
            snapshot, self.recovery_info = ResetterEnvCeres.get_random_reference_snapshot(self)
            state = self.reset_and_restore(snapshot=snapshot)
        self.update_ineq_matrices(state)
        self.n_recovery_steps = 0
        return state
    
    def init_recovery(self):
        required_base_env_attrs = ['max_recovery_steps', 'info_key_failure', 'info_key_success']
        for _k in required_base_env_attrs:
            assert hasattr(self, _k), 'Undefined attribute {0} in base environment within {1}'.format(_k, type(self))
        self.recovery_reward_failure = -self.max_recovery_steps

