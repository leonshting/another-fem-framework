# coding: utf-8

import numpy as np

from common.polynom_factory import glob_points, glob_weights
from interpolant.helpers import *


class Cell2PointsConverter:
    def __init__(self, **kwargs):

        #checking correctness of arguments
        checkup_success, checkup_msg, args_presented = self._check_constructor_arguments(**kwargs)
        if not checkup_success:
            raise Exception(checkup_msg)

        self.orders = kwargs['orders']

        if args_presented['sizes']:
            self.cum_size = [kwargs['sizes'][0][0][0], kwargs['sizes'][0][-1][-1]]
            self.sizes = kwargs['sizes']
        else:
            self.cum_size = [0., 1.]
            self.sizes = [[(i / len(j), (i + 1) / len(j)) for i, val in enumerate(j)] for j in self.orders]
        self.lengths = [[j - i for i, j in size] for size in self.sizes]

        if not args_presented['type'] or kwargs['type'] == 'lobatto':
            self.points = [[glob_points(size=size_val, order=order_val) for order_val, size_val in (zip(size, order))]
                           for size, order in zip(self.orders, self.sizes)]

            self.weights = [[glob_weights(order=order_val) * length_val / 2
                             for order_val, size_val, length_val in (zip(size, order, length))]
                            for size, order, length in zip(self.orders, self.sizes, self.lengths)]
        else:
            # TODO: implement uniform
            raise NotImplementedError()

        self.points_squeezed = [merge_arrays_with_adjacency(*points_val) for points_val in self.points]
        self.weights_squeezed = [concat_arrays_with_adjacency(*weights_val) for weights_val in self.weights]

        self.diag_weights = [np.diag(weights_sq) for weights_sq in self.weights_squeezed]

    #TODO: scale sizes and weights to [0,1]
    def _scale_to_01(self):
        pass

    @staticmethod
    def _check_constructor_arguments(**kwargs):
        possible_arguments = ['orders', 'sizes', 'type']
        for arg in possible_arguments:
            if arg in kwargs and not is_iterable_of_iterable(kwargs[arg]):
                return False, 'Both arguments should be iterables of iterables'

        if 'orders' not in kwargs:
            return False, 'Orders not provided'

        if 'sizes' in kwargs and not (kwargs['sizes'][0][0][0] == kwargs['sizes'][1][0][0]
         and kwargs['sizes'][0][-1][-1] == kwargs['sizes'][1][-1][-1]):
            return False, 'Ends of cuts must match'

        if 'type' in kwargs and 'type' not in ['uniform', 'lobatto']:
            return False, 'Type can be either uniform of lobatto'

        return True, 'Everything is OK', {arg: (arg in kwargs) for arg in possible_arguments}

    def get_weights(self):
        return self.weights_squeezed

    def get_diag_weights(self):
        return self.diag_weights

    def get_points(self):
        return self.points_squeezed

    def get_weights_and_points(self):
        return self.diag_weights, self.points_squeezed
