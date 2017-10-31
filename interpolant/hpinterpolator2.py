from interpolant import opt, strict, cell_to_points
from interpolant.helpers import *
from common.custom_types_for_typechecking import *


class HPInterpolator:
    def __init__(self, orders: interpolator_orders, sizes: interpolator_sizes):
        self.c2p = cell_to_points.Cell2PointsConverter(orders=orders, sizes=sizes)
        self.opt_solver = None
        self.strict_solver = strict.StrictSolve(
            points=self.c2p.get_points(),
            weights=self.c2p.get_diag_weights(),
            verbose=True
        )
        self.state = self.strict_solver.state

    def fit_strict(self, method='sympy', orders_to_add: int=0):
        max_pow_kw = self._orders_to_add2risk(orders_to_add=orders_to_add)
        self.strict_solver.fit_strict(
            max_power=self.strict_solver.get_max_pow(**max_pow_kw),
            method=method
        )
        self.state = self.strict_solver.state

    def fit_opt(self, fit_on='chebyshev', orders_forward=1, constraints=True):
        interps = self.strict_solver.get_interpolants()
        weights = self.strict_solver.get_weight_matrices()

        self.opt_solver = opt.OptSolve(
            matrices=interps,
            points=self.c2p.get_points(),
            strict_power=self.strict_solver.get_set_max_pow(),
            weights_matrices=weights,
            verbose=True
        )
        self.opt_solver.fit_opt(
            constraints=constraints,
            tr_fun_option=fit_on,
            orders_forward=orders_forward,
            verbose=True
        )
        self.state = self.opt_solver.state

    def fit_opt_restart(self, tolerance=1e-3):
        self.fit_opt_restart(tolerance=tolerance)
        self.state = self.opt_solver.state

    def get_interpolants(self):
        if self.state == self.opt_solver.states['FIT_OPT_SUCCESS']:
            return {'I_lr': np.array(self.opt_solver.I_lr_subbed).astype(np.float64),
                    'I_rl': np.array(self.opt_solver.I_rl_subbed).astype(np.float64)}
        else:
            raise Exception('Cannot use this function now, because fitting state is {}'.format(self.state))

    def fit(self,strict_method='sympy',
            strict_orders_to_add=0,
            opt_fit_on='chebyshev',
            opt_orders_forward=1,
            opt_constraints=True):
        self.fit_strict(method=strict_method, orders_to_add=strict_orders_to_add)
        self.fit_opt(fit_on=opt_fit_on, orders_forward=opt_orders_forward, constraints=opt_constraints)

    @staticmethod
    def _orders_to_add2risk(orders_to_add):
        if orders_to_add not in [-1,0,1]:
            raise Warning("That couldn't have happened")
        else:
            if orders_to_add == -1:
                return {'unrisky': True, 'risky': False}
            elif orders_to_add == 0:
                return {'unrisky': False, 'risky': False}
            else:
                return {'unrisky': False, 'risky': True}
