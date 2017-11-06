#coding-utf8
from functools import reduce

from scipy.sparse import coo_matrix

from .custom_types_for_typechecking import *

import numpy as np


def convert_edge_to_coorwise(edge: edge_nD_type):
    return np.array(edge).astype(np.int).T


def edge_length(edge: edge_nD_type):
    cwise_converted = convert_edge_to_coorwise(edge)
    changing_coord = [i[0] != i[1] for i in cwise_converted].index(True)
    return abs(edge[1][changing_coord] - edge[0][changing_coord]), changing_coord


def scale_edges(edge: edge_nD_type, div=2, how='upscale'):
    if div != 2:
        raise NotImplementedError('only 2:1 meshes are implemented')

    l, ch_coord = edge_length(edge)
    cwise_converted = convert_edge_to_coorwise(edge)
    if how == 'upscale':
        ch_edge_coord = [(min(cwise_converted[ch_coord]) - l, max(cwise_converted[ch_coord])), (min(cwise_converted[ch_coord]), max(cwise_converted[ch_coord]) + l)]
    elif how == 'downscale':
        ch_edge_coord = [(min(cwise_converted[ch_coord]), max(cwise_converted[ch_coord]) - l // 2), (min(cwise_converted[ch_coord]) + l // 2, max(cwise_converted[ch_coord]))]
    else:
        raise Exception('how has to be `upscale` or `downscale`')

    def subs_subtuple(tup, subtuple, index):
        ret_tup = list(tup)
        ret_tup[index] = subtuple
        return tuple(ret_tup)

    ret_edges = [np.array(subs_subtuple(cwise_converted, e, ch_coord)).T for e in ch_edge_coord]
    return [incr_repr_edge(tuple([tuple(i) for i in e])) for e in ret_edges]


def incr_repr_edge(edge: edge_nD_type):
    if np.logical_and.reduce(np.less_equal(np.array(edge[0]), np.array(edge[1]))):
        return edge
    else:
        return (edge[1], edge[0])


def is_pow2(num: int):
    return num!=0 and ((num & (num-1)) == 0)


def is_between_incl(val, borders):
    return val <= max(borders) and val >= min(borders)


def is_between(val, borders):
    return val < max(borders) and val > min(borders)


def on_edge_incl(point: vertex_nD_type, edge: edge_nD_type):
    l, ch_coord = edge_length(edge)
    coords_unchanged = [i for i in range(len(point)) if i != ch_coord]
    coords_unchanged_equal = reduce(lambda x,y: x and y, [e == v for e,v in zip([edge[0][i] for i in coords_unchanged], [point[i] for i in coords_unchanged])])
    return coords_unchanged_equal and point[ch_coord] >= edge[0][ch_coord] and is_between_incl(point[ch_coord], (edge[0][ch_coord], edge[1][ch_coord]))


def on_edge(point: vertex_nD_type, edge: edge_nD_type):
    l, ch_coord = edge_length(edge)
    coords_unchanged = [i for i in range(len(point)) if i != ch_coord]
    coords_unchanged_equal = reduce(lambda x,y: x and y, [e == v for e,v in zip([edge[0][i] for i in coords_unchanged], [point[i] for i in coords_unchanged])])
    return coords_unchanged_equal and point[ch_coord] > edge[0][ch_coord] and is_between(point[ch_coord], (edge[0][ch_coord], edge[1][ch_coord]))


def renorm_tuple(tup: Tuple, vice_norm: Tuple, new_norm: Tuple):
    return tuple([i*n//v for i,n,v in zip(tup, new_norm, vice_norm)])


def distributed_eye(pairtuples, shape):
    d_loc = [d[0] for d in pairtuples]
    d_glob = [d[1] for d in pairtuples]
    return coo_matrix(([1] * len(pairtuples), (d_loc, d_glob)), shape=shape)


def distributed_eye_easy(pairtuples, axis2shape):
    return distributed_eye(pairtuples=pairtuples, shape=(len(pairtuples), axis2shape))
