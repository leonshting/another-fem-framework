from scipy import optimize
from sympy import lambdify, Matrix

from .helpers import trace_on_points
from .verbose_printer import VerbosePrinter
import numpy as np


class OptSolve(VerbosePrinter):
    def __init__(self, matrices, weights_matrices, points, strict_power, **kwargs):
        super().__init__()
        self._states_setup()

        self.I_rl = matrices['I_RL']
        self.I_lr = matrices['I_LR']
        self.W_l = weights_matrices['L']
        self.W_r = weights_matrices['R']
        self.free_symbols = kwargs.get('free_symbols', self.I_lr.free_symbols)

        self.strict_power = strict_power
        self.points = points
        self.cum_size = [self.points[0][0], self.points[0][-1]]

        self.state = self.states['SET']

    @staticmethod
    def _check_trial_functions_option(option):
        return True if option in ['polynomial', 'chebyshev', 'custom'] else False

    def _states_setup(self):
        self.states = {
            'SET': 'Entry parameters are set',
            'FIT_OPT_SUCCESS': 'Fit-opt success',
            'FIT_OPT_FAILED': 'Fit-opt failed',
            'FIT_OPT_INTERRUPTED': 'Fit-opt interrupted',
        }

    def restart_opt(self, tolerance=1e-6, **kwargs):
        if self.state == self.states['FIT_OPT_INTERRUPTED']:
            try:
                res = self._fit(
                    self._set_constraints,
                    self._trial_functions,
                    initial_guess=self.opt_fit_result,
                    tolerance=tolerance,
                    minimize_options=self._set_minimize_options
                )
                if res:
                    self.state = self.states['FIT_OPT_SUCCESS']
                else:
                    self.state = self.states['FIT_OPT_FAILED']

                self._verbose_print(self.state)
            except KeyboardInterrupt:
                print(
                    "Results are saved, get your interploants in I_rl and I_lr fields or restart with higher tolerance")
                self.state = self.states['FIT_OPT_INTERRUPTED']
            finally:
                self.I_rl_subbed = self.I_rl.subs({i: j for i, j in zip(self.free_symbols, self.opt_fit_result)})
                self.I_lr_subbed = self.I_lr.subs({i: j for i, j in zip(self.free_symbols, self.opt_fit_result)})

                self.I_rl_subbed_symm = 0.5 * (self.I_rl_subbed + self.I_rl_subbed[::-1, ::-1])
                self.I_lr_subbed_symm = 0.5 * (self.I_lr_subbed + self.I_lr_subbed[::-1, ::-1])
        else:
            raise Exception('Cannot restart now, state is {}'.format(self.state))

    def fit_opt(self, constraints=True, tr_fun_option='chebyshev', **kwargs):
        self._set_constraints = constraints
        self._set_orders_forwards = kwargs.get('orders_forward', 2)
        self._set_minimize_options = kwargs.get('minimize_options', {'method': 'SLSQP'})

        if self._check_trial_functions_option(tr_fun_option):
            self.tr_fun_option = tr_fun_option
        else:
            raise Exception('trial functions may be either polynomial (default) or chebyshev or custom')
        if tr_fun_option == 'custom':
            raise NotImplementedError
        else:
            self._trial_functions = self._construct_trial_functions(
                self.strict_power,
                orders_forward=self._set_orders_forwards,
                trial_functions_type=self.tr_fun_option)

        try:
            res=self._fit(
                self._set_constraints,
                self._trial_functions,
                initial_guess=kwargs.get('initial_guess', np.random.randn(len(self.free_symbols))),
                minimize_options=self._set_minimize_options
            )
            if res:
                self.state = self.states['FIT_OPT_SUCCESS']
            else:
                self.state = self.states['FIT_OPT_FAILED']
            self._verbose_print(self.state)
        except KeyboardInterrupt:
            print("Results are saved, get your interploants in I_rl and I_lr fields or restart with higher tolerance")
            self.state = self.states['FIT_OPT_INTERRUPTED']
        finally:
            self.I_rl_subbed = self.I_rl.subs({i: j for i, j in zip(self.free_symbols, self.opt_fit_result)})
            self.I_lr_subbed = self.I_lr.subs({i: j for i, j in zip(self.free_symbols, self.opt_fit_result)})

            self.I_rl_subbed_symm = 0.5 * (self.I_rl_subbed + self.I_rl_subbed[::-1, ::-1])
            self.I_lr_subbed_symm = 0.5 * (self.I_lr_subbed + self.I_lr_subbed[::-1, ::-1])

    def _fit(self, constraints, trial_functions, initial_guess, minimize_options, **kwargs):
        f_lrrl = lambda x: 1 - max(
            np.abs(np.linalg.eigvals(
                lambdify(args=self.free_symbols,
                         expr=0.25 * (self.I_lr + self.I_lr[::-1, ::-1]) * (self.I_rl + self.I_rl[::-1, ::-1]),
                         modules='numpy')(*x))))
        f_rllr = lambda x: 1 - max(
            np.abs(np.linalg.eigvals(
                lambdify(args=self.free_symbols,
                         expr=0.25 * (self.I_rl + self.I_rl[::-1, ::-1]) * (self.I_lr + self.I_lr[::-1, ::-1]),
                         modules='numpy')(*x))))

        gll_points_traces = [
            [trace_on_points(points=points_sq, function=f) for points_sq in self.points] for
            f in trial_functions]

        epss_l = [self.I_rl * Matrix(gll_points_trace[1]) - Matrix(gll_points_trace[0]) for gll_points_trace in
                  gll_points_traces]
        epss_r = [self.I_lr * Matrix(gll_points_trace[0]) - Matrix(gll_points_trace[1]) for gll_points_trace in
                  gll_points_traces]

        objs_l = [(eps_l.T * self.W_l * eps_l)[0, 0] for eps_l, eps_r in zip(epss_l, epss_r)]
        objs_r = [(eps_r.T * self.W_r * eps_r)[0, 0] for eps_l, eps_r in zip(epss_l, epss_r)]

        objective = sum([obj_r + obj_l for obj_r, obj_l in zip(objs_l, objs_r)])

        self._verbose_print('Optimization started')

        f_obj = lambda x: lambdify(self.free_symbols, objective, 'numpy')(*x)
        self._f_objective = f_obj

        init = initial_guess
        if constraints:
            res = optimize.minimize(f_obj, init,
                                        constraints=(
                                            {'type': 'ineq', 'fun': f_lrrl},
                                            {'type': 'ineq', 'fun': f_rllr}),
                                        options={'disp': self._verbose,
                                                 'ftol': kwargs.get('tolerance', 1e-6)},
                                        callback=self._opt_callback(f_obj),
                                        tol=kwargs.get('tolerance', 1e-6),
                                        method=minimize_options['method'])
        else:
            res = optimize.minimize(f_obj, init,
                                        options={'disp': self._verbose,
                                                 'ftol': kwargs.get('tolerance', 1e-6)},
                                        callback=self._opt_callback(f_obj),
                                        tol=kwargs.get('tolerance', 1e-6),
                                        method=minimize_options['method'])

        self.opt_fit_result = res['x']
        return res['success']

    def _construct_trial_functions(self, power, trial_functions_type, orders_forward=2):
        ret_funcs = []
        if trial_functions_type == 'chebyshev':
            for p in range(power, power + orders_forward):
                mask = np.zeros(p + 1)
                mask[p] = 1
                for root_point in [0,1]:
                    ret_funcs.append(lambda x: np.polynomial.Chebyshev(coef=mask, domain=self.cum_size)(x-root_point))
            return ret_funcs
        elif trial_functions_type == 'polynomial':
            for p in range(power, power + orders_forward):
                ret_funcs.append(lambda x: x**p)
            return ret_funcs

    def _opt_callback(self, f_obj):
        def loss(xk):
            self.opt_fit_result = xk
            print('Loss: {}'.format(f_obj(xk)),
                  end='\r')

        return loss

    def _set_initial_guess(self):
        pass
