# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import gym
from ceres.tools.math import QPSolverQuadprog
import numpy as np
import tensorflow as tf

class ConstrainedEnv(gym.Env):
    '''
    Base class for constrained environments, with action correction prior to playing
    '''

    return_zero_if_opt_fails = True
    constraint_violation_factor = 1.
    ineq_vec_margin = 0.
    has_ineq = True
    has_eq   = True

    def __init__(self, *args, **kwargs):
        self.init_solver()
        self.check_instance()
        self.ineq_mat = None
        self.ineq_vec = None

    def init_solver(self):
        '''
        Correct actions with quadratic programming, define functions to implement within child classes
        '''
        self.correct_in_env = True
        self.required_functions = ['update_ineq_matrices']
        if self.has_ineq:
            self.required_functions.append('update_ineq_matrices')
        if self.has_eq:
            self.required_functions.append('update_eq_matrices')
        self.solver = QPSolverQuadprog()

    def check_instance(self):
        '''
        Check necessary attributes from parent and child classes
        '''
        for f_name in self.required_functions:
            assert hasattr(self, f_name), 'Required function {0} is not implemented in {1}'.format(f_name, type(self))
        for attr_name in ['observation_space', 'action_space']:
            assert hasattr(self, attr_name), 'Undefined attribute {0}: make sure the base environment is initialized'.format(attr_name)
        n_obs = self.observation_space.shape[0]
        if hasattr(self, 'n_obs'):
            assert self.n_obs == n_obs, 'Found two different values of n_obs: {0} and {1}'.format(self.n_obs, n_obs)
        else:
            self.n_obs = n_obs
        n_act = self.action_space.shape[0]
        if hasattr(self, 'n_act'):
            assert self.n_act == n_act, 'Found two different values of n_act: {0} and {1}'.format(self.n_act, n_act)
        else:
            self.n_act = n_act

    def set_ineq_margin(self, margin_param, relative=True):
        '''
        Define a margin that corrected actions must preserve w.r.t. constraints, that is, solve G x <= h - margin
        '''
        self.ineq_vec_margin = margin_param
        if relative:
            ac_space_pm = [0.5*(high-low) for low, high in zip(self.action_space.low, self.action_space.high)]
            self.ineq_vec_margin *= min(ac_space_pm)
        assert self.ineq_vec_margin >= 0, 'Negative margin not supported, but you can disable this check to allow constraint violation'


    def update_solver(self, do_update_eq=True, do_update_ineq=True, do_update_obj=True):
        '''
        Update QP solver parameters
        '''
        self.solver.reset(do_reset_eq=do_update_eq, do_reset_ineq=do_update_ineq, do_reset_obj=do_update_obj)
        self.solver.add_obj(self.obj_mat, self.obj_vec)
        if self.has_ineq:
            ineq_vec_solve = self.ineq_vec - self.ineq_vec_margin # account for conservative margin
            self.solver.add_ineq(self.ineq_mat, ineq_vec_solve)
        if self.has_eq:
            self.solver.add_eq(self.eq_mat, self.eq_vec)
        self.solver.update()

    def correct_action(self, target_action, do_update_eq=True, do_update_ineq=True, do_update_obj=True):
        '''
        Correct action by solving the QP and compute how much the uncorrected action violates the constraints
        '''
        # Only rebuild objective function matrices since inequality matrices are already rebuilt at the end of each step
        if do_update_obj:
            self.update_obj_matrices(target_action)
        self.update_solver(do_update_eq=do_update_eq, do_update_ineq=do_update_ineq, do_update_obj=do_update_obj)
        corrected_action, success = self.solver.solve()
        if success:
            corrected_action = np.reshape(corrected_action, target_action.shape)
        else:
            if self.return_zero_if_opt_fails:
                corrected_action = np.zeros(target_action.shape)
            else:
                corrected_action = target_action
        viol = self.calc_constraint_violation(target_action)
        return corrected_action, success, viol

    def print_ineq(self, ineq_mat=None, ineq_vec=None):
        '''
        Print inequality constraints in a human-readable format
        '''
        print(self.ineq_to_str(ineq_mat=ineq_mat, ineq_vec=ineq_vec))

    def ineq_to_str(self, ineq_mat=None, ineq_vec=None):
        '''
        Build a human-readable string for inequality constraints
        '''
        if ineq_mat is None:
            ineq_mat = self.ineq_mat
        if ineq_vec is None:
            ineq_vec = self.ineq_vec
        ineq_mat_str = str(ineq_mat)
        ineq_mat_lines_str = ineq_mat_str.split('\n')
        ineq_vec_str = str(ineq_vec)
        ineq_vec_lines_str = ineq_vec_str.split('\n')
        n_ineq = len(ineq_mat_lines_str)
        n_digits_max = len(str(n_ineq))
        opt_var_str = [['X{0}'.format(str(_i).zfill(n_digits_max))] for _i in range(n_ineq)]
        opt_var_lines_str = str(np.array(opt_var_str)).split('\n')
        ineq_str_lines = []
        for _i, (ineq_mat_line_str, opt_var_line_str, ineq_vec_line_str) in enumerate(zip(ineq_mat_lines_str, opt_var_lines_str, ineq_vec_lines_str)):
            ineq_str_line = '{0}{3} {1} {3}<= {2}{3}'.format(ineq_mat_line_str, opt_var_line_str, ineq_vec_line_str, ' ' if _i < n_ineq-1 else '')
            ineq_str_lines.append(ineq_str_line)
        ineq_str = '\n'.join(ineq_str_lines)
        return ineq_str

    def calc_constraint_violation(self, raw_action):
        '''
        Compute the L2 norm of the constraint violation margin for the uncorrected action
        '''
        if self.ineq_mat is not None:
            a = np.reshape(raw_action, (self.n_act, 1))
            ineq_diff = np.dot(self.ineq_mat, a) - self.ineq_vec
            ineq_val = np.maximum(ineq_diff, 0.)
            ineq_viol = np.linalg.norm(ineq_val)
        else:
            ineq_viol = 0.
        return ineq_viol

    def update_obj_matrices(self, target_action):
        '''
        Build objective function matrices of the form 1/2 xT P x + qT x,
        to minimize the distance between optimal and uncorrected (target) action, 1/2 || target - x || ^2,
        hence P = identity and q = -target
        '''
        self.obj_mat = np.eye(self.n_act)
        self.obj_vec = -np.reshape(target_action, (self.n_act, 1))

