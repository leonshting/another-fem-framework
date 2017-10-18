from operator_assembler.base_ddof_allocator import BaseAllocator2D


class AssemblyInterface2D:
    def __init__(self, allocator: BaseAllocator2D):
        self.allocator = allocator

    def iterate_cell_ddofs(self):
        for nl, cell in self.allocator.grid_interface.iterate_cells_fstb():
            ret_ddof_list = []
            for edge in cell.iterate_edges():
                ret_ddof_list.append(self.allocator.get_ddof_edge(cell, edge))
            yield ret_ddof_list