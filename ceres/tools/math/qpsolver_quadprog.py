# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import numpy as np
import quadprog
from .qpsolver import QPSolver

class QPSolverQuadprog(QPSolver):
    '''
    A class interfacing with the Quadprog QP solver
    '''

    def __init__(self, n_var=None, verbose=False):
        super(QPSolverQuadprog, self).__init__(n_var=n_var, verbose=verbose)

    def update_solver_specific(self):
        self.obj_mat_quadprog = self.obj_mat
        self.obj_vec_quadprog = -np.squeeze(self.obj_vec, axis=1)
        self.obj_mat_quadprog = self.obj_mat_quadprog.astype(dtype=np.float64)
        self.obj_vec_quadprog = self.obj_vec_quadprog.astype(dtype=np.float64)
        has_eq_loc   = self.eq_mat is not None
        has_ineq_loc = self.ineq_mat is not None
        if has_ineq_loc:
            if has_eq_loc:
                self.constraint_mat_quadprog = -np.vstack([self.eq_mat, self.ineq_mat]).transpose()
                self.constraint_vec_quadprog = -np.hstack([np.squeeze(self.eq_vec, axis=1), np.squeeze(self.ineq_vec, axis=1)])
            else:
                self.constraint_mat_quadprog = -self.ineq_mat.transpose()
                self.constraint_vec_quadprog = -np.squeeze(self.ineq_vec, axis=1)
        else:
            if has_eq_loc:
                self.constraint_mat_quadprog = -self.eq_mat.transpose()
                self.constraint_vec_quadprog = -np.squeeze(self.eq_vec, axis=1)
            else:
                self.constraint_mat_quadprog = None
                self.constraint_vec_quadprog = None
        if has_eq_loc or has_ineq_loc:
            self.constraint_mat_quadprog = self.constraint_mat_quadprog.astype(dtype=np.float64)
            self.constraint_vec_quadprog = self.constraint_vec_quadprog.astype(dtype=np.float64)

    def solve(self):
        try:
            self.solver_out = quadprog.solve_qp(self.obj_mat_quadprog, self.obj_vec_quadprog,
                                                self.constraint_mat_quadprog, self.constraint_vec_quadprog,
                                                self.n_eq)
            self.optimum = self.solver_out[0]
            self.success = True
        except ValueError as e:
            print('WARNING: solver failed ({0})'.format(e))
            self.optimum = np.zeros(self.n_var)
            self.success = False
        return self.optimum, self.success

if __name__ == '__main__':
    '''
    Implement example from cvxopt.org
    minimize 2 x1^2 + x2^2 + x1*x2 + x1 + x2
    subject to:
        x1 >= 0
        x2 >= 0
        x1 + x2 = 1
    '''
    qp_solver = QPSolverQuadprog()
    Q = 2.*np.array([[2., 0.5],
                     [0.5, 1.]])
    p = np.array([[1.],
                  [1.]])
    G1 = np.array([[-1., 0.]])
    h1 = np.array([[0.]])
    G2 = np.array([[0., -1.]])
    h2 = np.array([[0.]])
    A = np.array([[1., 1.]])
    b = np.array([[1.]])
    qp_solver.add_obj(Q, p)
    qp_solver.add_eq(A, b)
    qp_solver.add_ineq(G1, h1)
    qp_solver.add_ineq(G2, h2)
    qp_solver.update()
    x_opt, success = qp_solver.solve()
    print(x_opt, success)
    
    if input('Enter debug mode? y/[N]\n').lower() == 'y':
        import ipdb; ipdb.set_trace()
