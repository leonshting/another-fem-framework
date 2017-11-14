# coding: utf-8

import numpy as np
import itertools
from collections import Iterable
from sympy import preorder_traversal, Float
from scipy import linalg, compress, transpose
from matplotlib import pyplot as plt

def concat_arrays_with_adjacency(*arrays):
    ## numpy arrays as input
    ## stitching last position of n_th array and first one of n+1_th array
    l = len(arrays)
    ret_length = sum([f.shape[0] for f in arrays]) - (l - 1)
    ret_array = np.zeros(ret_length)
    pos0 = 0

    for arr in arrays:
        ret_array[pos0: pos0 + len(arr)] += arr
        pos0 = pos0 + len(arr) - 1
    return ret_array


def merge_arrays_with_adjacency(*arrays):
    ## numpy arrays as input
    ## stitching last position of n_th array and first one of n+1_th array
    l = len(arrays)
    ret_length = sum([f.shape[0] for f in arrays]) - (l - 1)
    ret_array = np.zeros(ret_length)
    pos0 = 0

    for arr in arrays:
        ret_array[pos0: pos0 + len(arr)] = arr
        pos0 = pos0 + len(arr) - 1
    return ret_array


## find every non-empty intersection
def intersection(cut1, cut2):
    f_slices = [max, min]
    T_slices = [[i, j] for i, j in zip(cut1, cut2)]
    retcut = tuple([f_slice(T_slice) for T_slice, f_slice in zip(T_slices, f_slices)])
    return retcut if retcut[0] < retcut[1] else None


def intersections(sizes, orders=None, with_belonging=False):
    # if orders presented - yields also max_order on relevant intersection

    intersections = []
    belongings = []
    retorders = []
    for i, j in itertools.product(*[enumerate(size) for size in sizes]):
        intersections.append(intersection(i[1], j[1]))
        belongings.append((i[0], j[0]))
        if orders is not None:
            retorders.append(max(orders[0][i[0]], orders[1][j[0]]))
    mask = [i is not None for i in intersections]
    return tuple([list(itertools.compress(i, mask)) for i in [intersections, belongings, retorders]])


def function_index(orders):
    index = []
    for order_pack in orders:
        index_pack = []
        cnt = 0
        for order in order_pack:
            index_pack.append(list(range(cnt, cnt + order + 1)))
            cnt += order
        index.append(index_pack)
    return index


def round_expression(expr, tol=5):
    ret_expr = expr
    for k in preorder_traversal(ret_expr):
        if isinstance(k, Float):
            ret_expr = ret_expr.subs(k, round(k, tol))
    return ret_expr

def null(A, eps=1e-6):
    u, s, vh = linalg.svd(A)
    np.save('sing_val', arr=s)
    padding = max(0,np.shape(A)[1]-np.shape(s)[0])
    null_mask = np.concatenate(((s <= eps), np.ones((padding,),dtype=bool)),axis=0)
    print(null_mask)
    null_space = compress(null_mask, vh, axis=0)
    return transpose(null_space)

def is_iterable_of_iterable(instance):
    correct = isinstance(instance, Iterable)
    if not correct:
        return False
    for i in instance:
        correct = correct and isinstance(i, Iterable)
    return correct

def trace_on_points(points, function):
    return np.vectorize(function)(np.array(points))


def convergence_test(I_rl, I_lr, p_l, p_r, init_size, tr_f, div_lim=5, weights=None):
    l = init_size[1] - init_size[0]
    f, axs = plt.subplots(ncols=2, nrows=div_lim - 1, figsize=(15, 15))
    epss_l = []
    epss_r = []
    for divi in range(1, div_lim):

        I_lr_new = linalg.block_diag(*(divi * [I_lr]))
        I_rl_new = linalg.block_diag(*(divi * [I_rl]))
        weights_new = [linalg.block_diag(*(divi * [weight])) for weight in weights]
        p_l_new = np.array(list(itertools.chain(*[(i * l / divi + p_l / divi) for i in range(0, divi)])))
        p_r_new = np.array(list(itertools.chain(*[(i * l / divi + p_r / divi) for i in range(0, divi)])))

        if weights is not None:
            eps_l2_r = np.dot(np.dot(
                tr_f(p_r_new) - np.dot(I_lr_new, tr_f(p_l_new)),
                weights_new[1] * l / divi), tr_f(p_r_new) - np.dot(I_lr_new, tr_f(p_l_new)))
            eps_l2_l = np.dot(np.dot(
                tr_f(p_l_new) - np.dot(I_rl_new, tr_f(p_r_new)),
                weights_new[0] * l / divi), tr_f(p_l_new) - np.dot(I_rl_new, tr_f(p_r_new)))
            epss_l.append(eps_l2_l)
            epss_r.append(eps_l2_r)
        axs[divi - 1][0].plot(p_l_new, tr_f(p_l_new))
        axs[divi - 1][0].plot(p_r_new, np.dot(I_lr_new, tr_f(p_l_new)))

        axs[divi - 1][1].plot(p_r_new, tr_f(p_r_new))
        axs[divi - 1][1].plot(p_l_new, np.dot(I_rl_new, tr_f(p_r_new)))
    return epss_l, epss_r


def explosion_test(I_rl, I_lr, p_l, p_r, tr_f, reps=5):
    f, axs = plt.subplots(ncols=2, nrows=1, figsize=(15, 5))
    new_tr_f_r = tr_f(p_r)
    new_tr_f_l = tr_f(p_l)
    for rep in range(reps):
        axs[0].plot(p_r, np.dot(np.dot(I_lr, I_rl), tr_f(p_r)))
        axs[1].plot(p_l, np.dot(np.dot(I_rl, I_lr), tr_f(p_l)))

        new_tr_f_r = np.dot(np.dot(I_lr, I_rl), new_tr_f_r)
        new_tr_f_l = np.dot(np.dot(I_rl, I_lr), new_tr_f_l)
