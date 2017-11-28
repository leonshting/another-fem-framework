#coding - utf8
import numpy as np

from common.custom_types_for_typechecking import *
from common.polynom_factory import gen_nodes
from operator_assembler.base_ddof_allocator import BaseAllocator2D


class GridDomain:

    def __init__(self, integer_size: size_nD_type, domain_size: domain_size_nD_type):
        if len(integer_size) == len(domain_size):
            self.dim = len(integer_size)
        else:
            raise Exception('Dimensions of size tuples must match.')

        self.integer_size = integer_size
        self.domain_size = domain_size
        self.dist_dict = {'lobatto': 'globatto', 'uniform': 'uniform'}
        self.pointwise_index = {}
        self.normal_index = {}
        self._ddof_cnt = 0

    def map_domain_to_grid(self, domain_point: domain_vertex_nD_type):
        """:return integer multi index of closest vertex"""
        return (np.array(self.integer_size) * (np.array(domain_point[0]) - np.array(self.domain_size[0])) /
                (np.array(self.domain_size[1]) - np.array(self.domain_size[0]))).astype(np.int)

    def map_grid_to_domain(self, grid_point: vertex_nD_type):
        return tuple([d_size * i / i_size
                      for i, i_size, d_size in zip(grid_point, self.integer_size, self.domain_size)])

    def make_pointwise_index(self, allocator: BaseAllocator2D):
        self._ddof_cnt = allocator.ddof_cnt
        for num_layer, cell in allocator.grid_interface.iterate_cells_fstb():
            order, dist = allocator.get_cell_props(cell)
            nodes_unscaled = gen_nodes(
                order=order,
                dim=2,
                size=tuple((i,i+j) for i,j in zip(cell.ll_vertex, cell.size)),
                distribution=self.dist_dict[dist]
            )
            for node, dof in zip(nodes_unscaled, allocator.get_cell_list_of_ddofs(cell)):
                node_scaled = self.map_grid_to_domain(node)
                if self.pointwise_index.get(node_scaled) is None:
                    self.pointwise_index[node_scaled] = [dof[1]]
                else:
                    self.pointwise_index[node_scaled].append(dof[1])
        self.pointwise_index = {k:set(v) for k,v in self.pointwise_index.items()}
        for k, v in self.pointwise_index.items():
            for dof in v:
                self.normal_index[dof] = k


    def vectorize_function(self, f):
        vec = np.zeros(self._ddof_cnt)
        for k,v in self.pointwise_index.items():
            for dof in v:
                vec[dof] = f(*k)
        return vec

    def devectorize_vector(self, vector):
        points_values = {}
        for k,v in self.normal_index.items():
            points_values[v] = vector[k]
        return points_values

    def devectorize_function(self, f):
        points_values = {}
        for k, v in self.normal_index.items():
            points_values[v] = f(*v)
        return points_values

