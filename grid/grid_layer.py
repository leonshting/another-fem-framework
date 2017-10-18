# coding - utf8
from typing import List

from common.custom_types_for_typechecking import *
from grid.cell import Cell2D


# TODO: probably make it abc.abstract
# TODO: proprtize


class GridLayer():

    def __init__(self, ll_vertices: List[vertex_nD_type], layer_number: int, index: List[int], div_index: List[int]):
        self.vertices = ll_vertices
        self.layer_num = layer_number
        self.rock_index = index
        self.div_index = div_index

        self.num_of_vertices, self.n2v_index = self._num2vertex_index()
        self.num_of_vertices, self.v2n_index = self._vertex2num_index()

    def _infer_cell_size(self):
        return 2**self.layer_num

    def _vertex2num_index(self):
        index = {num: v for num,v in enumerate(self.vertices)}
        return len(index), index

    def _num2vertex_index(self):
        index = {v: num for num,v in enumerate(self.vertices)}
        return len(index), index

    def get_num_of_cells(self):
        return len(self.vertices)


class GridLayer2D(GridLayer):

    def __init__(self, *args, **kwargs):
        super(GridLayer2D, self).__init__(*args, **kwargs)
        self.dim = 2
        self.cell_size = self._infer_cell_size()
        self._set_edge_index()
        self._set_vertex_index()

    def _set_vertex_index(self):
        self._vertex_index = {}
        for cell in self.iterate_cells():
            for vertex in cell.iterate_vertices():
                if self._vertex_index.get(vertex) is not None:
                    self._vertex_index[vertex].append(cell.ll_vertex)
                else:
                    self._vertex_index[vertex] = [cell.ll_vertex]

    def _set_edge_index(self):
        self._edge_index = {}
        for cell in self.iterate_cells():
            for edge in cell.iterate_edges():
                if self._edge_index.get(edge) is not None:
                    self._edge_index[edge].append(cell.ll_vertex)
                else:
                    self._edge_index[edge] = [cell.ll_vertex]

    def get_cells_by_edge(self, edge: edge_2D_type):
        got_cells = self._edge_index.get(edge)
        if got_cells is not None:
            return [Cell2D(size=self.cell_size, ll_vertex=i) for i in got_cells]
        else:
            return None

    def get_cells_by_vertex(self, vertex: vertex_2D_type):
        got_cells = self._vertex_index.get(vertex)
        if got_cells is not None:
            return [Cell2D(size=self.cell_size, ll_vertex=i) for i in got_cells]
        else:
            return None

    def _infer_cell_size(self):
        return tuple([super(GridLayer2D, self)._infer_cell_size()] * self.dim)

    def iterate_cells(self, return_index=False):
        for index, vertex in enumerate(self.vertices):
            if return_index:
                yield index, Cell2D(size=self.cell_size, ll_vertex=vertex)
            else:
                yield Cell2D(size=self.cell_size, ll_vertex=vertex)
