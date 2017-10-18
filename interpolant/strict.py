# coding: utf-8

from sympy import Matrix, Symbol, solve, linear_eq_to_matrix
from .helpers import round_expression, null
from .verbose_printer import VerbosePrinter
from scipy import linalg

import itertools
import numpy as np


class StrictSolve(VerbosePrinter):
    def __init__(self, points, weights, **kwargs):
        super().__init__(**kwargs)

        self._states_setup()
        self.total_points = [len(p_val) for p_val in points]
        self.weights = weights
        self.points = points

        self._symbolic_setup()
        self.state = self.states['PREFIT_STRICT']

    def _states_setup(self):
        self.states = {
            'SET': 'Entry parameters are set',
            'PREFIT_STRICT': 'Pre-fit strict',
            'FIT_STRICT_SUCCESS': 'Fit-strict success',
            'FIT_STRICT_FAILED': 'Strict part is badly configured',
            'PREFIT_OPT': 'Pre-fit opt'
        }

    def _symbolic_setup(self):
        self.symbols_for_M = [[Symbol('x_{}{}'.format(i, j)) for i in range(self.total_points[0])] for j in
                              range(self.total_points[1])]
        self.W_l = Matrix(self.weights[0])
        self.W_r = Matrix(self.weights[1])
        self.I_lr = Matrix(self.symbols_for_M)
        self.I_rl = self.W_l.inv() * self.I_lr.T * self.W_r

    def fit_strict(self, max_power, method='sympy', **kwargs):
        # TODO: svd variant (not implemented) or sympy solve variant and verbosity as a decorator

        self._max_power = max_power
        if self.state != self.states['PREFIT_STRICT']:
            raise Exception('Wrong order of the operations')
        points_pow_matrix = np.array([np.power(self.points, i) for i in range(max_power)])

        eqs_list = []
        for i in range(max_power):
            eqs_list.append(self.I_lr * Matrix(points_pow_matrix[i][0]) - Matrix(points_pow_matrix[i][1]))
            eqs_list.append(self.I_rl * Matrix(points_pow_matrix[i][1]) - Matrix(points_pow_matrix[i][0]))
        fl_eqs_list = []
        for mat in eqs_list:
            for idrow in range(mat.shape[0]):
                fl_eqs_list.append(round_expression(mat[idrow]))

        self._verbose_print('Number of equations: {}\n'
                            'Number of degrees of freedom: {}\n'
                            'Strictly stitched_power: {}\n'
                            'Start solving strict part'
                            .format(len(fl_eqs_list), len(self.I_lr.free_symbols), max_power - 1))

        if method == 'sympy':
            success, self.free_symbols = self._sympy_solve(fl_eqs_list, list(itertools.chain(*self.symbols_for_M)))
        elif method == 'svd':
            success, self.free_symbols = self._svd_solve(
                fl_eqs_list, 
                list(itertools.chain(*self.symbols_for_M)),
                eps=kwargs.get('options', {'eps': 1e-7}).get('eps', 1e-7))
        else:
            success = False

        if success:
            self.state = self.states['FIT_STRICT_SUCCESS']
            self._verbose_print('End solving strict part\nNullspaceRank: {}'.format(len(self.free_symbols)))
        else:
            self.state = self.states['FIT_STRICT_FAILED']
            self._verbose_print(self.state)

    def _sympy_solve(self, eqs_list, var_list):
        answer = solve(eqs_list, var_list)

        self.I_lr = self.I_lr.subs(answer).applyfunc(round_expression)
        self.I_rl = self.I_rl.subs(answer).applyfunc(round_expression)

        free_symbols = self.I_lr.free_symbols
        return len(answer) != 0, free_symbols

    def _svd_solve(self, eqs_list, var_list, eps=1e-7):
        M, rhs = [np.array(i).astype(np.float64) for i in linear_eq_to_matrix(eqs_list, var_list)]
        M_NS = null(M, eps=eps)
        some_solution, resid, rank, sigma = linalg.lstsq(M, rhs)

        self.aux_symbols = [Symbol('a_{}'.format(i)) for i in range(M_NS.shape[1])]
        aux_vector = Matrix(self.aux_symbols)
        self._verbose_print('Nullspace shape: {}'.format(M_NS.shape))
        answer = {var: (Matrix(some_solution) + Matrix(M_NS) * aux_vector)[i, 0] for i, var in enumerate(var_list)}
        self.I_lr = self.I_lr.subs(answer).applyfunc(round_expression)
        self.I_rl = self.I_rl.subs(answer).applyfunc(round_expression)

        free_symbols = self.I_lr.free_symbols
        return M_NS.shape[1] != 0, free_symbols



    def get_max_pow(self, risky=False):
        return (self.total_points[0] * self.total_points[1]) // (sum(self.total_points)) + int(risky)

    def get_set_max_pow(self):
        return self._max_power

    def get_weight_matrices(self):
        return {'R': self.W_r, 'L': self.W_l}

    def get_interpolants(self):
        if self.state == self.states['FIT_STRICT_SUCCESS']:
            return {'I_RL': self.I_rl, 'I_LR': self.I_lr}
        else:
            raise Exception('Not fitted - {}'.format(self.state))