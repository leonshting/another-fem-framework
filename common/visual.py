from matplotlib import pyplot as plt
import numpy as np
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import Axes3D

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



