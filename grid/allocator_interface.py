#coding-utf8
from typing import List, Generator

from common.custom_types_for_typechecking import *
from grid.cell import Cell2D
from grid.grid_layer import GridLayer2D
from grid.grid_manager import GridManager

cell_yield_type_2D = TypeVar('cell_yield', Tuple[Cell2D], Tuple[int, Cell2D])


class InterfaceDofAllocator2D:
    def __init__(self, grid_manager: GridManager, **kwargs):
        self.layers = grid_manager.grid_layers
        self.num_layers = len(self.layers)
        self.dim = 2

        self._default_order = kwargs.get('default_order', 4)
        self._default_dist = kwargs.get('default_dist', 'lobatto')

    def query_adj_cells_by_edge(self, cell: Cell2D, edge: edge_2D_type, num_layer: int, size_rel_filter=None):
        if num_layer == 0:
            join_adj_dict = {1: 'bigger', 0: 'same'}
        elif num_layer == self.num_layers - 1:
            join_adj_dict = {0: 'same', -1: 'smaller'}
        else:
            join_adj_dict = {0: 'same', 1: 'bigger', -1: 'smaller'}
        if size_rel_filter is not None:
            join_adj_dict = {k:v for k,v in join_adj_dict.items() if v in size_rel_filter}
        adj_cells_edge2edge = {}
        for tag, layer_candidate in [(alias, self.layers[num_layer + nl]) for nl, alias in join_adj_dict.items()]:
            for edge_candidate in cell.adj_edge_variants(edge)[tag]:
                if layer_candidate.get_cells_by_edge(edge_candidate) is not None:
                    adj_cells_edge2edge[(edge, edge_candidate)] = [c_cell for c_cell in layer_candidate.get_cells_by_edge(edge_candidate) if c_cell != cell]
                    if len(adj_cells_edge2edge[(edge, edge_candidate)]) == 0:
                        del adj_cells_edge2edge[(edge, edge_candidate)]
                    else:
                        adj_cells_edge2edge[(edge, edge_candidate)] = adj_cells_edge2edge[(edge, edge_candidate)][0]
        return adj_cells_edge2edge

    def query_adj_cells_by_vertex(self, cell: Cell2D, vertex: vertex_2D_type, num_layer: int):
        if num_layer == 0:
            join_adj_dict = {1: 'bigger', 0: 'same'}
        elif num_layer == self.num_layers - 1:
            join_adj_dict = {0: 'same', -1: 'smaller'}
        else:
            join_adj_dict = {0: 'same', 1: 'bigger', -1: 'smaller'}
        adj_cells_v2v = {}
        for tag, layer_candidate in [(alias, self.layers[num_layer + nl]) for nl, alias in join_adj_dict.items()]:
            for vertex_candidate in cell.iterate_vertices():
                if layer_candidate.get_cells_by_vertex(vertex_candidate) is not None:
                    adj_cells_v2v[vertex] = [c_cell for c_cell in layer_candidate.get_cells_by_vertex(vertex_candidate) if c_cell != cell]
                    if len(adj_cells_v2v[vertex]) == 0:
                        del adj_cells_v2v[vertex]
        return adj_cells_v2v

    def get_order_for_cell(self, cell: Cell2D):
        return self._default_order

    def get_dist_for_cell(self, cell: Cell2D):
        return self._default_dist

    def iterate_cells_fstb(self, yield_layer_num=True) -> cell_yield_type_2D:
        """iterates cells layer by layer from small to big"""
        for layer in self.layers:
            for cell in layer.iterate_cells():
                ret_list = []
                if yield_layer_num:
                    ret_list.append(layer.layer_num)
                ret_list.append(cell)
                yield tuple(ret_list)

    def iterate_cells_fbts(self, yield_layer_num=True) -> cell_yield_type_2D:
        """iterates cells layer by layer from small to big"""
        for layer in self.layers[::-1]:
            for cell in layer.iterate_cells():
                ret_list = []
                if yield_layer_num:
                    ret_list.append(layer.layer_num)
                ret_list.append(cell)
                yield tuple(ret_list)

    def get_cell_props(self, cell: Cell2D):
        return self.get_order_for_cell(cell), self.get_dist_for_cell(cell)