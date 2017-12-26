from typing import Tuple
import numpy as np
from copy import deepcopy
from scipy.ndimage import convolve
from skimage.measure import block_reduce
from itertools import product

from common.factorization import factor


class CoarseOnIndex:

    def __init__(self, refine_on_interfaces_with_disc=True, small_first=True):
        self._additional_refine_control = refine_on_interfaces_with_disc
        self._small_first = small_first

    @staticmethod
    def get_kernel_size(cell_size: Tuple):
        return tuple(map(lambda x: x + 2, cell_size))

    @staticmethod
    def corners_indices_iterator(dim: int =2):
        return product([0, -1], repeat=dim)

    @staticmethod
    def faces_indices_iterator(dim: int=2):
        indices = [slice(0, None) for i in range(dim)]
        for i in range(dim):
            indices_subbed = deepcopy(indices)
            indices_subbed[i] = [0, -1]
            yield indices_subbed

    @classmethod
    def get_kernel(cls, kernel_size: Tuple, corner_zero: bool=True):
        kernel = np.zeros(kernel_size)
        for face_ix in cls.faces_indices_iterator(dim=len(kernel_size)):
            kernel[face_ix] = 1
        if corner_zero:
            for corner_ix in cls.corners_indices_iterator(dim=len(kernel_size)):
                kernel[corner_ix] = 0
        return kernel

    @staticmethod
    def get_cell_sizes_for_coarsening(data_shape: Tuple):
        # works with equal number of factors
        data_shape_factorized = [factor(i) for i in data_shape]
        return [k for k in zip(*data_shape_factorized)]

    @staticmethod
    def get_slices_for_conv_product(cell_size: Tuple):
        return [slice(None, None, i) for i in cell_size]

    @staticmethod
    def get_conv_origin(kernel_size: Tuple):
        kernel_center = tuple(map(lambda x: (x - 1) // 2, kernel_size))
        return tuple(map(lambda x: (x[1] - x[0]), zip(np.ones(len(kernel_size)), kernel_center)))

    def _additional_coarse_control(self, kernel, data_coarsed, slice_for_conv, to_compare):
        prev_layer_same_rock = np.isclose(convolve(
            input=data_coarsed.astype(np.int64), weights=kernel / kernel.sum(),
            origin=self.get_conv_origin(kernel_size=kernel.shape), mode='reflect')[slice_for_conv], to_compare[0])
        return prev_layer_same_rock

    def coarse_on_index(self, index: np.array):
        layer_counter = 0
        divided = [np.ones(index.shape, dtype=np.bool)]
        data_coarsed = [index]
        coarsed_to_layer = []

        cell_sizes = self.get_cell_sizes_for_coarsening(data_shape=index.shape)

        for num, cell_size in enumerate(cell_sizes):
            kernel = self.get_kernel(kernel_size=self.get_kernel_size(cell_size=cell_size))
            slice_for_conv = self.get_slices_for_conv_product(cell_size=cell_size)

            prev_layer_divided = (convolve(
                input=divided[num].astype(np.int8), weights=kernel,
                origin=self.get_conv_origin(kernel_size=kernel.shape),
                cval=1, mode='constant')[slice_for_conv] == np.sum(kernel).astype(np.int8))

            # 0 - windowed max, 1 - windowed mean
            to_compare = [block_reduce(block_size=cell_size, image=data_coarsed[num], func=f) for f in
                          [np.max, np.mean]]

            # additional coarsening helper
            if self._additional_refine_control:
                prev_layer_same_rock = self._additional_coarse_control(kernel, data_coarsed[num],
                                                                       slice_for_conv, to_compare)
                current_divided = np.logical_and(np.logical_and(prev_layer_divided, prev_layer_same_rock),
                                             (to_compare[0] == to_compare[1]))
            else:
                current_divided = np.logical_and(prev_layer_divided, (to_compare[0] == to_compare[1]))

            divided.append(current_divided)
            coarsed_to_layer.append((divided[num] - np.kron(current_divided, np.ones(cell_size))).astype(np.bool))
            data_coarsed.append(to_compare[1])
            layer_counter += 1
        return {
            'layer_counter': layer_counter,
            'coarsed_to_layer': coarsed_to_layer,
            'cell_sizes': cell_sizes,
        }