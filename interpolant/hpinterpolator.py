from collections import Iterable

from scipy import optimize
from sympy import *

from common.polynom_factory import glob_points, glob_weights
from .helpers import *


class HPInterpolator:
    def __init__(self, **kwargs):

        self.states_enum = {
            'SET': 0,
            'PREFIT_STRICT': 1,
            'PREFIT_OPT': 2,
            'FIT_STRICT_FAILED': 3,
            'SUCCESS': 4,
            'OPT_FIT_FAILED': 5,
            'OPT_FIT_INTERRUPTED': 6,
        }

        self.states = ['Entry parameters are set',
                       'Pre-fit strict',
                       'Pre-fit opt',
                       'Strict part is badly configured',
                       'Interpolants are ready',
                       'Opt part is not successful',
                       'Opt part interrupted']

        self.verbose = True if kwargs.get('verbose', False) else False

        if len(kwargs) == 0:
            raise Exception('Use explicit constructors: uniform_cuts_constructor, etc')

        _sizes_p_ = 'sizes' in kwargs.keys()
        _orders_p_ = 'orders' in kwargs.keys()

        if _orders_p_:
            if not self._orders_structure_checkup(kwargs['orders']):
                raise Exception('orders must be a list of lists')
            self.orders = kwargs['orders']
        else:
            raise Exception('orders must be provided')

        if _sizes_p_:
            if not self._sizes_structure_checkup(kwargs['sizes']):
                raise Exception('sizes must be a list of lists')
            self.cum_size = self._cum_size(sizes=kwargs['sizes'])
            self.sizes = kwargs['sizes']
        else:
            self.cum_size = [0., 1.]
            self.sizes = [[(i / len(j), (i + 1) / len(j)) for i, val in enumerate(j)] for j in self.orders]

        self.state = self.states[self.states_enum['SET']]

        self._setup()
        self._symbolic_setup()

    def _setup(self):
        self.functions_index = function_index(self.orders)

        self.total_points = [sum(order_val) + 1 for order_val in self.orders]

        self.lengths = [[j - i for i, j in size] for size in self.sizes]

        self.gll_points = [[glob_points(size=size_val, order=order_val) for order_val, size_val in (zip(size, order))]
                           for size, order in zip(self.orders, self.sizes)]

        self.gll_weights = [[glob_weights(order=order_val) * length_val / 2
                             for order_val, size_val, length_val in (zip(size, order, length))]
                            for size, order, length in zip(self.orders, self.sizes, self.lengths)]

        self.gll_points_squeezed = [merge_arrays_with_adjacency(*gll_points_val) for gll_points_val in self.gll_points]
        self.gll_weights_squeezed = [np.diag(concat_arrays_with_adjacency(*gll_weights_val)) for gll_weights_val in
                                     self.gll_weights]

    def _symbolic_setup(self):
        self.symbols_for_M = [[Symbol('x_{}{}'.format(i, j)) for i in range(self.total_points[0])] for j in
                              range(self.total_points[1])]
        self.I_lr = Matrix((self.symbols_for_M))

        self.W_l = Matrix(self.gll_weights_squeezed[0])
        self.W_r = Matrix(self.gll_weights_squeezed[1])

        # express each and every unknown in a single matrix - R2L projection
        self.I_rl = self.W_l.inv() * self.I_lr.T * self.W_r

        self.state = self.states[self.states_enum['PREFIT_STRICT']]

    def fit_strict(self, max_power, method='sympy'):
        ## TODO: svd variant (not implemented) or sympy solve variant and verbosity as a decorator

        if self.state != self.states[self.states_enum['PREFIT_STRICT']]:
            raise Exception('Wrong order of the operations')

        MAX_POW = max_power
        gll_points_pow_matrix = np.array([np.power(self.gll_points_squeezed, i) for i in range(MAX_POW)])

        # gll_points_pow_matrix: first index - order, second - LR
        #                                                      01

        eqs_list = []
        for i in range(MAX_POW):
            eqs_list.append(self.I_lr * Matrix(gll_points_pow_matrix[i][0]) - Matrix(gll_points_pow_matrix[i][1]))
            eqs_list.append(self.I_rl * Matrix(gll_points_pow_matrix[i][1]) - Matrix(gll_points_pow_matrix[i][0]))

        fl_eqs_list = []
        for mat in eqs_list:
            for idrow in range(mat.shape[0]):
                fl_eqs_list.append(round_expression(mat[idrow]))

        self._verbose_print('Number of equations: {}\nNumber of degrees of freedom: {}\nStart solving strict part'
                            .format(len(fl_eqs_list), len(self.I_lr.free_symbols)))

        if method == 'sympy':
            success, self.free_symbols = self._sympy_solve(fl_eqs_list, list(itertools.chain(*self.symbols_for_M)))

        elif method == 'svd':
            raise NotImplementedError()

        else:
            success = False

        if success:
            self.state = self.states[self.states_enum['PREFIT_OPT']]
            self._verbose_print('End solving strict part\nNullspaceRank: {}'.format(len(self.free_symbols)))
        else:
            self.state = self.states[self.states_enum['FIT_STRICT_FAILED']]

    ## todo: bug if double restart
    def restart_opt(self, tolerance=1e-6, **kwargs):
        if self.state == self.states[self.states_enum['OPT_FIT_INTERRUPTED']]:
            self.fit_opt(trial_functions=self._construct_trial_functions(self._set_MAX_POW,
                                                                         orders_forward=kwargs.get('orders_forward',
                                                                                                   self._set_orders_forward)),
                         constraints=self._set_constraints,
                         initial_guess=self.opt_fit_result_int,
                         tolerance=tolerance)
            self._print_opt_results_msg()

    def fit_opt(self, trial_functions, constraints=True, **kwargs):

        f_LRRL = lambda x: 1 - max(
            np.abs(np.linalg.eigvals(lambdify(args=self.free_symbols, expr=self.I_lr * self.I_rl)(*x))))
        f_RLLR = lambda x: 1 - max(
            np.abs(np.linalg.eigvals(lambdify(args=self.free_symbols, expr=self.I_rl * self.I_lr)(*x))))

        # gll_points_trace = [np.array(trial_functions[0](gll_points_sq))
        #                    for gll_points_sq in self.gll_points_squeezed]

        gll_points_traces = [
            [self.trace_on_points(points=gll_points_sq, function=f) for gll_points_sq in self.gll_points_squeezed] for
            f in trial_functions]

        epss_l = [self.I_rl * Matrix(gll_points_trace[1]) - Matrix(gll_points_trace[0]) for gll_points_trace in
                  gll_points_traces]
        epss_r = [self.I_lr * Matrix(gll_points_trace[0]) - Matrix(gll_points_trace[1]) for gll_points_trace in
                  gll_points_traces]

        objs_l = [(eps_l.T * self.W_l * eps_l)[0, 0] for eps_l, eps_r in zip(epss_l, epss_r)]
        objs_r = [(eps_r.T * self.W_r * eps_r)[0, 0] for eps_l, eps_r in zip(epss_l, epss_r)]

        objective = sum([round_expression(obj_r + obj_l) for obj_r, obj_l in zip(objs_l, objs_r)])

        self._verbose_print('Optimization started')

        f_obj = lambda x: lambdify(self.free_symbols, objective)(*x)
        self._f_objective = f_obj
        # max eigval way

        ##TODO rewrite in fashion
        try:
            init = kwargs.get('initial_guess', np.random.randn(len(self.free_symbols)))
            if constraints:
                res = optimize.minimize(f_obj, init,
                                        constraints=(
                                            {'type': 'ineq', 'fun': f_LRRL},
                                            {'type': 'ineq', 'fun': f_RLLR}),
                                        options={'disp': self.verbose,
                                                 'ftol': kwargs.get('tolerance', 1e-6)},
                                        callback=self._opt_callback(f_obj),
                                        tol=kwargs.get('tolerance', 1e-6),
                                        method='SLSQP')
            else:
                res = optimize.minimize(f_obj, init,
                                        options={'disp': self.verbose,
                                                 'ftol': kwargs.get('tolerance', 1e-6)},
                                        callback=self._opt_callback(f_obj),
                                        tol=kwargs.get('tolerance', 1e-6),
                                        method='SLSQP')

            self.opt_fit_result = res
            if res['success']:
                self.state = self.states[self.states_enum['SUCCESS']]
            else:
                self.state = self.states[self.states_enum['OPT_FIT_FAILED']]
        except:
            print("Results are saved, get your interploants in I_rl and I_lr fields or restart with higher tolerance")
            self.state = self.states[self.states_enum['OPT_FIT_INTERRUPTED']]
        finally:
            self.I_rl_subbed = self.I_rl.subs({i: j for i, j in zip(self.free_symbols, self.opt_fit_result_int)})
            self.I_lr_subbed = self.I_lr.subs({i: j for i, j in zip(self.free_symbols, self.opt_fit_result_int)})
    #to_impl
    def fit(self, constraints=True, risky_max_pow=False, **kwargs):

        MAX_POW = self._get_max_pow() + int(risky_max_pow)

        self._set_MAX_POW = MAX_POW
        self._set_orders_forward = kwargs.get('opt_fit_orders_forward', 2)
        self._set_constraints = constraints

        if self.state == self.states[self.states_enum['PREFIT_STRICT']]:
            self._verbose_print('Maximum strictly stitched power: {}'.format(MAX_POW - 1))
            self.fit_strict(MAX_POW, method=kwargs.get('strict_fit_method', 'sympy'))

        if self.state == self.states[self.states_enum['PREFIT_OPT']]:
            self.fit_opt(trial_functions=self._construct_trial_functions(MAX_POW, orders_forward=kwargs.get(
                'opt_fit_orders_forward', 2)),
                         constraints=constraints)

        self._print_opt_results_msg()
    # toimpl
    def _print_opt_results_msg(self):
        if self.state == self.states[self.states_enum['SUCCESS']]:
            print('OK ' + self.state)
            return True
        else:
            print('Bad ' + self.state)
            return False
    #to_--implement
    def get_interpolants(self):
        if self.state == self.states[self.states_enum['SUCCESS']]:
            return {'I_lr': np.array(self.I_lr_subbed).astype(np.float64),
                    'I_rl': np.array(self.I_rl_subbed).astype(np.float64)}
        else:
            raise Exception('Cannot use this function now, because fitting state is {}'.format(self.state))

    def _sympy_solve(self, eqs_list, var_list):

        answer = solve(eqs_list, var_list)

        self.I_lr = self.I_lr.subs(answer).applyfunc(round_expression)
        self.I_rl = self.I_rl.subs(answer).applyfunc(round_expression)

        free_symbols = self.I_lr.free_symbols

        return len(answer) != 0, free_symbols

    def svd_solve(self, eqs_list, var_list):
        M, rhs = [np.array(i).astype(np.float64) for i in linear_eq_to_matrix(eqs_list, var_list)]

    def _get_max_pow(self):
        ddof_number = self.total_points[0] * self.total_points[1]
        return ddof_number // (sum(self.total_points))

    def _get_number_of_eqs(self, power):
        return power * (sum(self.total_points))

    def _verbose_print(self, msg):
        if self.verbose:
            print(msg)

    @classmethod
    def uniform_cuts_constructor(cls, orders, **kwargs):
        return cls(orders=orders, **kwargs)

    @classmethod
    def non_uniform_cuts_constructor(cls, orders, sizes, **kwargs):
        return cls(orders=orders, sizes=sizes, **kwargs)

    @classmethod
    def _orders_structure_checkup(cls, orders):
        return cls._is_iterable_of_iterable(orders)
    #impl
    @staticmethod
    def trace_on_points(points, function):
        return np.vectorize(function)(np.array(points))

    @classmethod
    def _sizes_structure_checkup(cls, sizes):
        correct = cls._is_iterable_of_iterable(sizes)
        if not correct:
            return False
        else:
            for size in sizes:
                aux_bool = True
                for s_val, s_val_next in zip(size[:-1], size[1:]):
                    aux_bool = aux_bool and (s_val[-1] == s_val[0])
                correct = correct and aux_bool
            correct = correct and (sizes[0][0][0] == sizes[1][0][0] and sizes[0][-1][-1] == sizes[1][-1][-1])
            return correct

    # should be used only if sizes_strucutre is ok
    #done in constructor
    @staticmethod
    def _cum_size(sizes):
        return [sizes[0][0][0], sizes[0][-1][-1]]

    #impl
    def _opt_callback(self, f_obj):
        def loss(xk):
            print('Loss: {}'.format(f_obj(xk)))
            self.opt_fit_result_int = xk

        return loss

    #impl
    @staticmethod
    def _is_iterable_of_iterable(instance):
        correct = isinstance(instance, Iterable)
        if not correct:
            return False
        for i in instance:
            correct = correct and isinstance(i, Iterable)
        return correct

    #impl
    def _construct_trial_functions(self, power, orders_forward=2):
        ret_funcs = []
        for p in range(power, power + orders_forward):
            mask = np.zeros(p + 1)
            mask[p] = 1
            ret_funcs.append(np.polynomial.Chebyshev(coef=mask, domain=self.cum_size))
        return ret_funcs
