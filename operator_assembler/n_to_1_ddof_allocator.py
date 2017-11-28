from collections import defaultdict
from typing import Dict

from common.polynom_factory import gen_nodes
from grid.cell import Cell2D
from operator_assembler.base_ddof_allocator import BaseAllocator2D
from common.custom_types_for_typechecking import *


class Nto1Allocator2D(BaseAllocator2D):
    def __init__(self, *args, **kwargs):
        super(Nto1Allocator2D, self).__init__(*args, **kwargs)
        self.priority = ['size_match', 'match', 'border', 'smaller', 'bigger']

    def _make_ddof_index(self):
        for num_layer, cell in self.grid_interface.iterate_cells_fstb():
            adj_cells_stitching_modes = self._stitching_modes(num_layer=num_layer, cell=cell)
            for edge, adj_edge_cells, mode in adj_cells_stitching_modes:
                self._connect_edges(host_edge=edge, peer=adj_edge_cells, host_cell=cell, stitching_mode=mode)
            self._allocate_interior_ddofs(host_cell=cell)

    # TODO: convenience function to construct endpoint index

    def _make_conjugate_vertex_index(self):
        sizes = {}
        vertex_dofs = defaultdict(list)

        for (ll_vertex, vertex), (local_dof, global_dof) in self._vertex_ddof_index.items():
            sizes[ll_vertex] = max(sizes.get(ll_vertex, 0), max([abs(i - j) for i, j in zip(ll_vertex, vertex)]))
            vertex_dofs[vertex].append((ll_vertex, global_dof))

        self._conjugate_vertex_index = vertex_dofs
        self._sizes = sizes

    def _merge_ddof_in_index(self):
        cvi_replacement = defaultdict(list)
        id2v_replacement = {}

        for vertex, list_dofs_props in self._conjugate_vertex_index.items():
            ll_vertices = defaultdict(list)
            unique_dofs = {}
            for ll_vertex, global_dof in list_dofs_props:
                ll_vertices[self._sizes[ll_vertex]].append(ll_vertex)
                cvi_replacement[vertex].append((ll_vertex, global_dof))
                id2v_replacement[global_dof] = vertex
                unique_dofs[self._sizes[ll_vertex]] = (ll_vertex, global_dof)
            for size, ll_vertices in ll_vertices.items():
                for ll_vertex in ll_vertices:
                    self._vertex_ddof_index[(ll_vertex, vertex)] = (
                        self._vertex_ddof_index[(ll_vertex, vertex)][0],
                        unique_dofs[size][1]
                    )
            self._conjugate_vertex_index = cvi_replacement
            self._id_to_vertex_index = id2v_replacement

    def make_complete_index(self):
        self._make_ddof_index()
        self._make_conjugate_vertex_index()
        self._merge_ddof_in_index()

    @staticmethod
    def _get_stitching_mode(host_edge: edge_2D_type, peer_edges: Dict[Tuple, Cell2D], host_props, peer_props):
        if len(peer_edges) == 0:
            return 'border'
        elif len(peer_edges) == 1 and list(peer_edges.keys())[0][1] == host_edge:
            if host_props == peer_props[0]:
                return 'match'
            else:
                return 'size_match'
        elif len(peer_edges) > 1:
            return 'smaller'
        else:
            return 'bigger'

    def _weakly_connect_edges(self, host_edge: edge_2D_type, peer: Dict[Tuple, Cell2D], host_cell: Cell2D, how: str):

        host_props = {(host_cell.ll_vertex, host_edge):
                          self.grid_interface.get_cell_props(host_cell)}
        peer_props = {(p_cell.ll_vertex, p_edge[1]):
                          self.grid_interface.get_cell_props(p_cell) for p_edge, p_cell in peer.items()}

        host_list = {(host_cell.ll_vertex, host_edge):
                         self.get_flat_list_of_ddofs(host_edge, host_cell)}
        peer_list = {(p_cell.ll_vertex, p_edge[1]):
                         self.get_flat_list_of_ddofs(p_edge[1], p_cell) for p_edge, p_cell in peer.items()}

        if how in ['smaller', 'size_match']:
            self._weak_edge_connections[(host_cell.ll_vertex, host_edge)] = peer_list
            self._weak_edge_connections_props[(host_cell.ll_vertex, host_edge)] = peer_props

            for p_edge, p_cell in peer.items():
                peer_edge = p_edge[1]
                self._weak_edge_connections[(p_cell.ll_vertex, peer_edge)] = host_list
                self._weak_edge_connections_props[(p_cell.ll_vertex, peer_edge)] = host_props
        else:
            raise Exception('Program flow cannot turn here')

    def _stitching_modes(self, cell: Cell2D, num_layer: int):
        adj_cells_stitching_modes = []
        for edge in cell.iterate_edges():
            adj_edge_cells = self.grid_interface.query_adj_cells_by_edge(cell, edge, num_layer=num_layer)
            host_order, host_dist = self.grid_interface.get_cell_props(cell)
            peer_props = [self.grid_interface.get_cell_props(p_cell) for p_cell in adj_edge_cells.values()]

            stitching_mode = self._get_stitching_mode(
                edge, adj_edge_cells,
                host_props=(host_order, host_dist),
                peer_props=peer_props)
            adj_cells_stitching_modes.append((edge, adj_edge_cells, stitching_mode))

        return sorted(adj_cells_stitching_modes, key=lambda x: self.priority.index(x[2]))

    def _connect_edges(self, host_edge: edge_2D_type,
                       peer: Dict[Tuple, Cell2D],
                       host_cell: Cell2D,
                       stitching_mode: str):

        host_order, host_dist = self.grid_interface.get_cell_props(host_cell)
        peer_props = [self.grid_interface.get_cell_props(p_cell) for p_cell in peer.values()]

        normed_edge_host = host_cell.edge_normed_by_size(host_edge)
        host_cell_fe = self._get_fe_cell(size=host_cell.size, order=host_order, dist=host_dist)

        if stitching_mode in ['border', 'smaller', 'bigger', 'size_match']:
            self._allocate_local_ddofs_edge(edge=host_edge, cell=host_cell, cell_fe=host_cell_fe)
            if stitching_mode == 'size_match':
                for (adj_edge, peer_cell), p_props in zip(peer.items(), peer_props):
                    peer_edge = adj_edge[1]
                    peer_cell_fe = self._get_fe_cell(size=peer_cell.size, order=p_props[0], dist=p_props[1])
                    self._allocate_local_ddofs_edge(edge=peer_edge, cell=peer_cell, cell_fe=peer_cell_fe)
        else:
            for (adj_edge, peer_cell), p_props in zip(peer.items(), peer_props):
                peer_edge = adj_edge[1]
                peer_cell_fe = self._get_fe_cell(size=peer_cell.size, order=p_props[0], dist=p_props[1])
                ld_host_list = host_cell_fe.get_edge_ddof_index()[normed_edge_host]

                to_merge_with_edge = self._edge_ddof_index.get((peer_cell.ll_vertex, peer_edge))
                if to_merge_with_edge is not None:
                    self._edge_ddof_index[(host_cell.ll_vertex, host_edge)] = \
                        [(j, i[1]) for i, j in zip(to_merge_with_edge, ld_host_list)]
                else:
                    self._edge_ddof_index[(host_cell.ll_vertex, host_edge)] = \
                        [(i, self._ddof_cnt + num) for num, i in enumerate(ld_host_list)]
                    self._ddof_cnt += len(ld_host_list)
                for vertex_peer, vertex_host in zip(peer_edge, host_edge):
                    local_ddof_host = host_cell_fe.get_vertex_ddof_index()[host_cell.vertex_normed_by_size(vertex_host)]
                    local_ddof_peer = peer_cell_fe.get_vertex_ddof_index()[peer_cell.vertex_normed_by_size(vertex_host)]
                    to_merge_with_vertex = self._vertex_ddof_index.get(
                        (peer_cell.ll_vertex, vertex_peer),
                        self._vertex_ddof_index.get((host_cell.ll_vertex, vertex_host))
                    )
                    if to_merge_with_vertex is not None:
                        self._vertex_ddof_index[(host_cell.ll_vertex, vertex_host)] = \
                            (local_ddof_host, to_merge_with_vertex[1])
                        self._vertex_ddof_index[(peer_cell.ll_vertex, vertex_peer)] = \
                            (local_ddof_peer, to_merge_with_vertex[1])
                    else:
                        self._vertex_ddof_index[(host_cell.ll_vertex, vertex_host)] = \
                            (local_ddof_host, self._ddof_cnt)
                        self._vertex_ddof_index[(peer_cell.ll_vertex, vertex_peer)] = \
                            (local_ddof_peer, self._ddof_cnt)
                        self._ddof_cnt += 1
        if stitching_mode in ['smaller', 'size_match']:
            self._weakly_connect_edges(host_edge=host_edge,
                                       peer=peer,
                                       host_cell=host_cell,
                                       how=stitching_mode)
