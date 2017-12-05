# coding - utf8
from functools import reduce

import numpy as np


from common.helpers import is_pow2
from grid.grid_layer import GridLayer2D, GridLayer


class GridManager:
    # TODO: propertize
    # TODO: add options for refine process

    def __init__(self, dim: int = 2, method: str = 'coarse2to1', **kwargs):
        self.method_options = ['uniform', 'coarse2to1']

        self.dim = dim
        self.method = method
        self._alloc_constructors()
        # TODO: correct handling of kwargs

        self._max_coarsening_layer = kwargs.get('method_options', {'max_coarsening_layer': 100})\
            .get('max_coarsening_layer')

    def _alloc_constructors(self):
        if self.dim == 2:
            self.Layer = GridLayer2D

        if self.dim == 3:
            self.Layer = GridLayer

    def _check_args_before_fit(self, **kwargs):
        if self.method not in self.method_options:
            raise Exception('No such method')
        if self.method == 'coarse2to1':
            try:
                return self._check_data_2_to_1(kwargs['data'])
            except:
                print('Error in data validation for method coarse2to1')
        if self.method == 'uniform':
            return True

    def _check_data_2_to_1(self, data):
        dims_eq = len(data.shape) == self.dim
        valid = reduce(lambda x, y: x and y, [is_pow2(i) for i in data.shape])
        return valid and dims_eq

    def fit(self, data: np.array):
        self._check_args_before_fit(data=data)
        self.grid_layers = []
        if self.method == 'coarse2to1':
            from .refiners.coarse_on_index_2_to_1 import CoarseOnIndex
            gp = CoarseOnIndex(max_coarsening_layer=self._max_coarsening_layer)
            dict_data = gp.coarse_on_index(index=data)
            for layer_num, layer_size, layer_rock in zip(range(dict_data['layer_counter']),
                                                         dict_data['layer_scalers'],
                                                         dict_data['layer_rock_num']):
                div_index = [2] * len(layer_rock)
                self.grid_layers.append(self.Layer(layer_number=layer_num,
                                                   ll_vertices=list(layer_rock.keys()),
                                                   index=list(layer_rock.values()),
                                                   div_index=div_index))
        return self

    def draw_grid(self, file_to_save=None):
        if len(self.grid_layers):
            lines = []
            from matplotlib import pyplot as plt
            from matplotlib.collections import LineCollection
            for layer in self.grid_layers:
                for cell in layer.iterate_cells():
                    for edge in cell.iterate_edges():
                        lines.append(edge)
            fig, ax = plt.subplots(figsize=(15,15))
            lines = LineCollection(lines)
            ax.add_collection(lines)

            ax.autoscale()
            plt.show()
            if file_to_save is not None:
                fig.savefig(file_to_save)
