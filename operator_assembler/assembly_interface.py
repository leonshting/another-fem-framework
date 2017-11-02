import itertools
from operator import itemgetter

from grid.cell import Cell2D
from operator_assembler.base_ddof_allocator import BaseAllocator2D


class AssemblyInterface2D:
    def __init__(self, allocator: BaseAllocator2D):
        self.allocator = allocator

    #strange shit maybe delete
    def _get_ddofs_for_cell(self, cell):
        ddof_loc2glob_list = []
        for edge in cell.iterate_edges():
            ddof_loc2glob_list.append(self.allocator.get_ddof_edge(cell, edge))
        ddof_loc2glob_list.append(self.allocator.get_ddof_int(cell))
        ddof_loc2glob_list = list(itertools.chain(*ddof_loc2glob_list))
        for vertex in cell.iterate_vertices():
            ddof_loc2glob_list.append(self.allocator.get_ddof_vertex(cell, vertex))
        ret_list = [i[1] for i in sorted(ddof_loc2glob_list, key=itemgetter(0))]
        return ret_list

    # strange shit maybe delete
    def iterate_cell_ddofs(self, yield_cell=True):
        for nl, cell in self.allocator.grid_interface.iterate_cells_fstb():
            ret_ddofs = self._get_ddofs_for_cell(cell)
            if yield_cell:
                yield cell, ret_ddofs
            else:
                yield ret_ddofs

    def iterate_weak_connections(self, yield_cell=True):
        for nl, cell in self.allocator.grid_interface.iterate_cells_fstb():
            if yield_cell:
                yield cell, self.allocator.get_weakly_connected_edges(cell)
            else:
                yield self.allocator.get_weakly_connected_edges(cell)

    def iterate_ddofs_and_wconn(self, yield_cell=True, how='fbts', yield_smaller_neighbors_cell=True):
        iterator = self.allocator.grid_interface.iterate_cells_fbts
        if how == 'fstb':
            iterator = self.allocator.grid_interface.iterate_cells_fstb
        elif how != 'fbts':
            raise Exception('how can be either fstb or fbts')
        for nl, cell in iterator():
            ret_ddofs = self.allocator.get_cell_list_of_ddofs(cell=cell)
            adj_cell_query_results = {}
            weak_connections = {}
            #implement another mode wiht yielding every ddof
            if yield_smaller_neighbors_cell:
                peers = self.allocator.get_weakly_connected_edges(cell)
                unique_edges = set([p[0] for p in peers])
                for peer in peers:
                    result = self.allocator.grid_interface.query_adj_cells_by_edge(
                        cell=cell,
                        edge=peer[0],
                        num_layer=nl,
                        size_rel_filter=['smaller'])
                    if len(result) != 0:
                        tmp = (peer[1], result[(peer[0], peer[1])])
                        if adj_cell_query_results.get(peer[0]) is None:
                            adj_cell_query_results[peer[0]] = [tmp]
                        else:
                            adj_cell_query_results[peer[0]].append(tmp)

                        if weak_connections.get(peer[0]) is None:
                            weak_connections[peer[0]] = [peer[1:]]
                        else:
                            weak_connections[peer[0]].append(peer[1:])


            if yield_cell:
                yield {'ddofs': ret_ddofs, #this one may be unneeded
                       'wconn': weak_connections,
                       'cell': cell,
                       'adj_cells': adj_cell_query_results
                       }
            else:
                yield {'ddofs': ret_ddofs,
                       'wconn': self.allocator.get_weakly_connected_edges(cell),
                       }

    def get_ddof_count(self):
        return self.allocator.ddof_cnt
