import itertools

from common.helpers import *
from common.custom_types_for_typechecking import *
from grid.cell import Cell2D


class FiniteElement2D(Cell2D):
    def __init__(self, order: int, distribution: str, *args, **kwargs):
        super(FiniteElement2D, self).__init__(*args, **kwargs)
        self.order = order
        self.distribution = distribution
        self.dim = 2
        self._ordered_deltas = [[i * j for i, j in zip(p, (self.order, self.order))] for p in self._edges_deltas]

        self._set_ddof_index()

    def iterate_vertices_normed_by_order(self):
        l_r_coordinates = tuple((i, j) for i, j in zip((0, 0), (self.order, self.order)))
        return itertools.product(*l_r_coordinates)

    def iterate_edges_normed_by_order(self):
        current_vertex = (0, 0)
        for delta in self._ordered_deltas:
            new_vertex = tuple([i + j for i, j in zip(current_vertex, delta)])
            yield incr_repr_edge((current_vertex, new_vertex))
            current_vertex = new_vertex

    def _set_ddof_index(self):
        num_ddofs = (self.order + 1) ** self.dim
        ddof_points = {}
        ddof_points_inv = {}
        ls = 2 * [range(self.order + 1)]
        for num, (x, y) in enumerate(itertools.product(*ls)):
            ddof_points[num] = (x, y)
            ddof_points_inv[(x, y)] = num
        vertex_ddofs = {v: ddof_points_inv[v] for v in self.iterate_vertices_normed_by_order()}
        edge_ddofs = {e: sorted([ddof for point, ddof in ddof_points_inv.items() if on_edge(point, e)]) for e in
                      self.iterate_edges_normed_by_order()}
        interior_ddofs = sorted([i for i in range(num_ddofs) if
                          i not in list(itertools.chain(*edge_ddofs.values())) and i not in vertex_ddofs.values()])

        vertex_ddofs_normed_by_size = {renorm_tuple(k, vice_norm=(self.order, self.order), new_norm=self.size): v for
                                       k, v in vertex_ddofs.items()}
        edge_ddofs_normed_by_size = {
        tuple([renorm_tuple(ku, vice_norm=(self.order, self.order), new_norm=self.size) for ku in k]): v for k, v in
        edge_ddofs.items()}
        self._vertex_ddofs = vertex_ddofs_normed_by_size
        self._edge_ddofs = edge_ddofs_normed_by_size
        self._interior_ddofs = interior_ddofs

    def get_ddof_index(self):
        return self._vertex_ddofs, self._edge_ddofs, self._interior_ddofs

    def get_interior_ddof_index(self):
        return self._interior_ddofs

    def get_edge_ddof_index(self):
        return self._edge_ddofs

    def get_vertex_ddof_index(self):
        return self._vertex_ddofs

    @classmethod
    def from_cell2d(cls, cell: Cell2D, order: int, distribution: str):
        return cls(order=order, distribution=distribution, **cell.__dict__)