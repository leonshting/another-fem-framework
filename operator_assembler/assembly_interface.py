import itertools
from operator import itemgetter

from grid.cell import Cell2D
from operator_assembler.base_ddof_allocator import BaseAllocator2D


class AssemblyInterface2D:
    def __init__(self, allocator: BaseAllocator2D):
        self.allocator = allocator

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

    def iterate_cell_ddofs(self, yield_cell = True):
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

    def iterate_ddofs_and_wconn(self, yield_cell=True):
        for nl, cell in self.allocator.grid_interface.iterate_cells_fstb():
            ret_ddofs = self._get_ddofs_for_cell(cell)
            if yield_cell:
                yield {'ddofs': ret_ddofs,
                       'wconn': self.allocator.get_weakly_connected_edges(cell),
                       'neighbor': self.allocator.get_weakly_connected_neighbor(cell),
                       'cell': cell,
                       'cell_props': self.allocator.get_cell_props(cell)}
            else:
                yield {'ddofs': ret_ddofs,
                       'wconn': self.allocator.get_weakly_connected_edges(cell),
                       'neighbor': self.allocator.get_weakly_connected_neighbor(cell)
                       }

    def get_ddof_count(self):
        return self.allocator.ddof_cnt