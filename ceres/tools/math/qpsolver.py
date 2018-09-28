# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import numpy as np

class QPSolver(object):
    '''
    A base class to interface with QP solvers
    '''

    def __init__(self, n_var=None, verbose=False):
        self.n_var = n_var
        self.verbose = verbose
        self.reset()

    def reset(self, do_reset_obj=True, do_reset_eq=True, do_reset_ineq=True):
        if do_reset_obj:
            self.reset_obj()
        if do_reset_eq:
            self.reset_eq()
        if do_reset_ineq:
            self.reset_ineq()

    def update(self):
        self.build_obj()
        self.build_eq()
        self.build_ineq()
        self.update_solver_specific()

    def reset_eq(self):
        self.eq_mat_list = []
        self.eq_vec_list = []
        self.eq_mat = None
        self.eq_vec = None
        self.n_eq = 0
        self.reset_eq_solver_specific()

    def reset_ineq(self):
        self.ineq_mat_list = []
        self.ineq_vec_list = []
        self.ineq_mat = None
        self.ineq_vec = None
        self.n_ineq = 0
        self.reset_ineq_solver_specific()

    def reset_obj(self):
        self.obj_mat_list = []
        self.obj_vec_list = []
        self.obj_mat = None
        self.obj_vec = None
        self.n_obj = 0
        self.reset_obj_solver_specific()

    def check_mat_vec(self, mat, vec):
        '''
        Ensure that mat and vec are numpy arrays and of appropriate dimensions
        '''
        mat = np.array(mat)
        vec = np.array(vec)
        if self.n_var is None:
            self.n_var = mat.shape[1]
        else:
            assert mat.shape[1] == self.n_var, 'Invalid constraint matrix size {0} for {1} variables'.format(mat.shape, self.n_var)
        assert mat.ndim == 2, 'Invalid constraint matrix dimensions: expected 2, got {0}'.format(mat.ndim)
        assert vec.ndim == 2, 'Invalid constraint vector dimensions: expected 2, got {0}'.format(vec.ndim)
        assert mat.shape[0] == vec.shape[0], 'Inconsistent constraint matrix and vector sizes'
        assert vec.shape[1] == 1, 'Invalid constraint vector size {0}, should have one column'.format(mat.shape)
        return mat, vec

    def add_obj(self, mat, vec, build=False):
        mat, vec = self.check_mat_vec(mat, vec)
        assert mat.shape[0] == mat.shape[1], 'Invalid objective matrix shape {0}, should be square'.format(mat.shape)
        self.obj_mat_list.append(mat)
        self.obj_vec_list.append(vec)
        if build:
            self.build_obj()

    def build_obj(self):
        self.n_obj = len(self.obj_mat_list)
        assert self.n_obj > 0
        self.obj_mat = sum(self.obj_mat_list)
        self.obj_vec = sum(self.obj_vec_list)
        self.build_obj_solver_specific()

    def add_eq(self, mat, vec, build=False):
        mat, vec = self.check_mat_vec(mat, vec)
        self.eq_mat_list.append(mat)
        self.eq_vec_list.append(vec)
        if build:
            self.build_eq()

    def build_eq(self):
        if len(self.eq_mat_list) > 0:
            self.eq_mat = np.concatenate(self.eq_mat_list, axis=0)
            self.eq_vec = np.concatenate(self.eq_vec_list, axis=0)
            self.n_eq = self.eq_mat.shape[0]
        else:
            self.eq_mat = None
            self.eq_vec = None
            self.n_eq = 0
        self.build_eq_solver_specific()

    def add_ineq(self, mat, vec, build=False):
        if (mat is None) or (vec is None):
            assert (mat is None) and (vec is None), 'Constraint incomplete: mat={0}, vec={1}'.format(mat, vec)
            return
        mat, vec = self.check_mat_vec(mat, vec)
        n_ineq_loc = mat.shape[0]
        if n_ineq_loc > 0:
            self.ineq_mat_list.append(mat)
            self.ineq_vec_list.append(vec)
            if build:
                self.build_ineq()

    def build_ineq(self):
        if len(self.ineq_mat_list) > 0:
            self.ineq_mat = np.concatenate(self.ineq_mat_list, axis=0)
            self.ineq_vec = np.concatenate(self.ineq_vec_list, axis=0)
            self.n_ineq = self.ineq_mat.shape[0]
        else:
            self.ineq_mat = None
            self.ineq_vec = None
            self.n_ineq = 0
        self.build_ineq_solver_specific()

    def reset_obj_solver_specific(self):
        pass

    def reset_eq_solver_specific(self):
        pass

    def reset_ineq_solver_specific(self):
        pass

    def build_obj_solver_specific(self):
        pass

    def build_eq_solver_specific(self):
        pass

    def build_ineq_solver_specific(self):
        pass

    def update_solver_specific(self):
        pass

    def solve(self):
        raise NotImplementedError()


