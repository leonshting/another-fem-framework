from matplotlib import pyplot as plt
import numpy as np
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import Axes3D
from common.tests import partial_diff_test

def plot_surface_unstructured_w_dict(point_val_dict, plot_domain_shape, int_domain_shape):
    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(111, projection='3d')

    xi = np.linspace(0, plot_domain_shape[0], int_domain_shape[0])
    yi = np.linspace(0, plot_domain_shape[1], int_domain_shape[1])
    Xs = [i[0] for i in point_val_dict.keys()]
    Ys = [i[1] for i in point_val_dict.keys()]
    zi = [i for i in point_val_dict.values()]

    Z = griddata(Xs, Ys, zi, xi, yi, interp='linear')

    X, Y = np.meshgrid(xi, yi)

    ax1.plot_surface(X, Y, Z)

    plt.show()


def plot_sparse_pattern(matrix):
    from matplotlib import pyplot as plt
    plt.spy(matrix, markersize=0.5)
    plt.show()


def deps_partial_diff(pointnum, matrix, num2point_index, print_text=False, p2n_index=None):
    span_x = 0.025
    span_y = 0.048
    nonzero_ind = matrix[pointnum].nonzero()

    printed = {i: False for i in nonzero_ind[1]}
    plt.axis('equal')
    for id in nonzero_ind[1]:
        if print_text and not np.isclose(matrix[pointnum, id], [0.0], atol=5e-7):
            if len(p2n_index[num2point_index[id]]) == 1:
                plt.text(num2point_index[id][0] + span_x,
                         num2point_index[id][1] + span_y,
                     s=str("%.3f"%matrix[pointnum, id]))
            else:

                l = len(p2n_index[num2point_index[id]])
                for ord, num in enumerate(p2n_index[num2point_index[id]]):
                    if num in nonzero_ind[1] and not printed[num]:
                        plt.text(num2point_index[id][0] + span_x,
                                 num2point_index[id][1] - span_y*(l-1)/2 + ord*span_y,
                                 s=str("%.3f"%matrix[pointnum, num]))
                        printed[num] = True
    plt.scatter(
        [num2point_index[i][0]+.01 for i in nonzero_ind[1] if not np.isclose(matrix[pointnum, i], [0.0], atol=5e-7)],
        [num2point_index[i][1] for i in nonzero_ind[1] if not np.isclose(matrix[pointnum, i], [0.0], atol=5e-7)],
        c=[['red', 'blue'][i == pointnum] for i in nonzero_ind[1] if not np.isclose(
            matrix[pointnum, i], [0.0], atol=5e-7)])
    plt.scatter([num2point_index[pointnum][0]], [num2point_index[pointnum][1]], c='blue')
