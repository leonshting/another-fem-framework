from itertools import chain
from operator import itemgetter
from typing import Dict

from common.custom_types_for_typechecking import *
from grid.allocator_interface import InterfaceDofAllocator2D
from grid.cell import Cell2D
from operator_assembler.fe_cell import FiniteElement2D


class BaseAllocator2D:
    def __init__(self, grid_interface: InterfaceDofAllocator2D):
        self.grid_interface = grid_interface
        self._fe_cells = {}

        self._edge_ddof_index = {}
        self._vertex_ddof_index = {}
        self._interior_ddof_index = {}
        self._weak_edge_connections = {}
        self._weak_edge_connections_props = {}

        ##conjugate index for vertices - thing to get dofs and cells that are/have vertex by vertex
        self._conjugate_vertex_index = {}
        self._id_to_vertex_index = {}

        self._sizes = {}
        self._ddof_cnt = 0

    def _get_fe_cell(self, size, order, dist):
        if self._fe_cells.get((size, order)) is None:
            self._fe_cells[(size, order)] = FiniteElement2D.from_cell2d(
                Cell2D(size=size, ll_vertex=(0,0)),
                order=order,
                distribution=dist)
        return self._fe_cells[(size, order)]

    def _allocate_interior_ddofs(self, host_cell: Cell2D):
        host_order = self.grid_interface.get_order_for_cell(host_cell)
        host_dist = self.grid_interface.get_dist_for_cell(host_cell)
        host_cell_fe = self._get_fe_cell(size=host_cell.size, order=host_order, dist=host_dist)

        to_insert = host_cell_fe.get_interior_ddof_index()
        self._interior_ddof_index[host_cell.ll_vertex] = \
            [(i, num + self._ddof_cnt) for num, i in enumerate(to_insert)]
        self._ddof_cnt += len(to_insert)

    def _allocate_local_ddofs_edge(self, edge: edge_2D_type, cell: Cell2D, cell_fe: FiniteElement2D):
        normed_edge_host = cell.edge_normed_by_size(edge)
        ld_list = cell_fe.get_edge_ddof_index()[normed_edge_host]
        self._edge_ddof_index[(cell.ll_vertex, edge)] = \
            [(i, self._ddof_cnt + num) for num, i in enumerate(ld_list)]
        self._ddof_cnt += len(ld_list)

        for vertex in edge:
            local_ddof = cell_fe.get_vertex_ddof_index()[cell.vertex_normed_by_size(vertex)]
            if self._vertex_ddof_index.get((cell.ll_vertex, vertex)) is None:
                self._vertex_ddof_index[(cell.ll_vertex, vertex)] = (local_ddof, self._ddof_cnt)
                self._ddof_cnt += 1

    def get_flat_list_of_ddofs(self, edge: edge_2D_type, cell: Cell2D):
        edge_ddofs = self._edge_ddof_index[(cell.ll_vertex, edge)]
        vertex_ddofs = [self._vertex_ddof_index[(cell.ll_vertex, host_vertex)] for host_vertex in edge]

        return [vertex_ddofs[0]] + edge_ddofs + [vertex_ddofs[1]]

    def get_flat_list_of_ddofs_global(self, edge: edge_2D_type, cell: Cell2D):
        return [i[1] for i in self.get_flat_list_of_ddofs(edge=edge, cell=cell)]

    def get_cell_list_of_ddofs(self, cell: Cell2D):
        ddof_list = []
        for edge in cell.iterate_edges():
            ddof_list.extend(self.get_ddof_edge(cell, edge))
        for vertex in cell.iterate_vertices():
            ddof_list.append(self.get_ddof_vertex(cell=cell, vertex=vertex))
        ddof_list.extend(self.get_ddof_int(cell=cell))
        return sorted(ddof_list, key=itemgetter(0))

    def get_cell_list_of_ddofs_global(self, cell:Cell2D):
        return [i[1] for i in self.get_cell_list_of_ddofs(cell)]

    def get_ddof_edge(self, cell: Cell2D, edge: edge_2D_type):
        return self._edge_ddof_index.get((cell.ll_vertex, edge))

    def get_ddof_vertex(self, cell: Cell2D, vertex: vertex_2D_type):
        return self._vertex_ddof_index.get((cell.ll_vertex, vertex))

    def get_ddof_int(self, cell: Cell2D):
        return self._interior_ddof_index.get(cell.ll_vertex)

    def get_weakly_connected_edges(self, cell: Cell2D):
        ret_list = []
        for edge in cell.iterate_edges():
            for k_edge, v_edge in self._weak_edge_connections.get((cell.ll_vertex, edge), {}).items():
                ret_list.append((edge, k_edge[1], tuple([i for i in sorted(v_edge, key=itemgetter(0))])))
        return ret_list

    def get_weakly_connected_neighbor(self, cell: Cell2D, return_self=True):
        ret_list = []
        for edge in cell.iterate_edges():
            for peer_k, peer_v in self._weak_edge_connections.get((cell.ll_vertex, edge), {}).items():
                for k, v in self._weak_edge_connections_props[peer_k].items():
                    if k[1] != edge:
                        ret_list.append((k[1], v))
        return ret_list

    def get_cell_props(self, cell):
        return self.grid_interface.get_cell_props(cell)

    @property
    def ddof_cnt(self):
        return self._ddof_cnt