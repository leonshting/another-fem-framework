# coding - utf8
import itertools

from common.custom_types_for_typechecking import *
from common.helpers import scale_edges, incr_repr_edge


class Cell:
    # TODO: propertize
    def __init__(self, size: size_nD_type, ll_vertex: vertex_nD_type, *args, **kwargs):
        if len(size) == len(ll_vertex):
            self.size = size
            self.ll_vertex = ll_vertex

        else:
            raise Exception('Dimensions of size and LL_vertex must match')

    def iterate_vertices(self):
        rr_vertex = tuple([i + j for i, j in zip(self.ll_vertex, self.size)])
        l_r_coordinates = tuple((i, j) for i, j in zip(self.ll_vertex, rr_vertex))
        return itertools.product(*l_r_coordinates)

    def iterate_vertices_normed_by_size(self):
        l_r_coordinates = tuple((i, j) for i, j in zip((0, 0), self.size))
        return itertools.product(*l_r_coordinates)

    def vertex_normed_by_size(self, vertex):
        return tuple([i-j for i, j in zip(vertex, self.ll_vertex)])

    @staticmethod
    def adj_edge_variants(edge: edge_nD_type):
        return {'same': [edge], 'bigger': scale_edges(edge, how='upscale'),
                'smaller': scale_edges(edge, how='downscale')}

    def __eq__(self, other):
        return self.ll_vertex == other.ll_vertex and self.size == self.size


class Cell2D(Cell):
    def __init__(self, *args, **kwargs):
        super(Cell2D, self).__init__(*args, **kwargs)
        self._edges_deltas = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        self._sized_deltas = [[i * j for i, j in zip(p, self.size)] for p in self._edges_deltas]

    def iterate_edges(self):
        current_vertex = self.ll_vertex
        for delta in self._sized_deltas:
            new_vertex = tuple([i + j for i, j in zip(current_vertex, delta)])
            yield incr_repr_edge((current_vertex, new_vertex))
            current_vertex = new_vertex

    def edge_normed_by_size(self, edge):
        return tuple([tuple([i - j for i, j in zip(vertex, self.ll_vertex)]) for vertex in edge])
