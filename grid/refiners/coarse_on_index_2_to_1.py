import numpy as np
from scipy import ndimage
import itertools


class CoarseOnIndex:

    # TODO: propertize

    def __init__(self, max_coarsening_layer: int=100):
        self.max_coarsening_layer = max_coarsening_layer

    @staticmethod
    def _get_2_to_1_kernel():
        kernel = np.zeros((4, 4))
        combs = [itertools.product(*comb) for comb in itertools.permutations([[1, 2], [0, -1]])]
        for comb in combs:
            for c in comb:
                kernel[c] = 1
        kernel_sum = np.sum(kernel).astype(np.int32)
        return kernel, kernel_sum

    @classmethod
    def _pool_it(cls, data, data_divided, balanced=True):
        kernel, kernel_sum = cls._get_2_to_1_kernel()

        shape = data.shape
        dim = len(shape)
        tmp_reshape = []
        for s, d in zip((np.array(shape) / 2).astype(np.int32), [2] * dim):
            tmp_reshape.append(s)
            tmp_reshape.append(d)
        data_reshaped = data.reshape(tmp_reshape)
        data_divided_reshaped = data_divided.reshape(tmp_reshape)
        pooling_shape = tuple((np.array(shape) / 2).astype(np.int32)) + tuple([2 ** dim])
        if dim == 3:
            data_rs_swapped = data_reshaped.swapaxes(1, 2).swapaxes(4, 2)
            data_d_rs_swapped = data_divided_reshaped.swapaxes(1, 2).swapaxes(4, 2)
            s = data_rs_swapped.reshape(*pooling_shape).sum(axis=-1)
            maxs = data_rs_swapped.reshape(*pooling_shape).max(axis=-1) * (2 ** dim)
            sdiv = np.logical_and.reduce(data_d_rs_swapped.reshape(*pooling_shape), axis=-1)
            if balanced:
                sdiv = np.logical_and((ndimage.convolve
                                       (data_divided.astype(np.int8), weights=kernel, mode='constant', cval=1)[::2, ::2,
                                       ::2] == kernel_sum), sdiv)
        elif dim == 2:
            data_rs_swapped = data_reshaped.swapaxes(1, 2)
            data_d_rs_swapped = data_divided_reshaped.swapaxes(1, 2)
            s = data_rs_swapped.reshape(*pooling_shape).sum(axis=-1)
            maxs = data_rs_swapped.reshape(*pooling_shape).max(axis=-1) * (2 ** dim)
            sdiv = np.logical_and.reduce(data_d_rs_swapped.reshape(*pooling_shape), axis=-1)
            if balanced:
                sdiv = np.logical_and((ndimage.convolve
                                       (data_divided.astype(np.int8), weights=kernel, mode='constant', cval=1)[::2,
                                       ::2] == kernel_sum), sdiv)
        else:
            raise Exception("Wrong dimension somehow")
        # scale utility
        this_level = np.logical_and(sdiv, (maxs == s))
        prev_level = np.logical_xor(np.kron(this_level, np.ones([2] * dim)), data_divided)
        #
        return (s / (2 ** dim)).astype(np.int32), this_level, prev_level

    def coarse_on_index(self, index: np.array):
        data = index
        max_layer = self.max_coarsening_layer

        layers = []
        layers_bool = []
        refinement_levels = []

        layer_rock_num = []

        layer_counter = 0
        layer_scalers = []

        layers.append(data)
        layers_bool.append(np.ones(data.shape, dtype=np.bool))
        while layer_counter < max_layer and \
                np.logical_or.reduce(layers_bool[-1].flatten()) and \
                        np.multiply.reduce(layers[-1].shape) != 1:

            rock_dict = {}

            new_layer_tuple = self._pool_it(layers[-1], layers_bool[-1], balanced=True)
            it_num_rock = np.nditer(layers[-1], flags=['multi_index'])
            it_is_refined = np.nditer(new_layer_tuple[2], flags=['multi_index'])

            while not (it_is_refined.finished):
                if (it_is_refined[0].item() or (
                        layer_counter == max_layer - 1 and layers_bool[-1][it_num_rock.multi_index])):
                    rock_dict[tuple([i * 2 ** (layer_counter) for i in it_num_rock.multi_index])] = (
                    it_num_rock[0].item())
                it_num_rock.iternext()
                it_is_refined.iternext()

            layers.append(new_layer_tuple[0])
            layers_bool.append(new_layer_tuple[1])
            refinement_levels.append(new_layer_tuple[2])

            layer_rock_num.append(rock_dict)
            layer_scalers.append(2 ** layer_counter)
            layer_counter += 1
        return {
            'layer_counter': layer_counter,
            'layers': layers,
            'layer_rock_num': layer_rock_num,
            'layer_scalers': layer_scalers,
            'refinement_levels': refinement_levels,
            'layers_bool': layers_bool,
        }