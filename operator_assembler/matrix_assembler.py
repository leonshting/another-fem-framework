# coding-utf8
import h5py
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from operator_assembler.assembly_interface import AssemblyInterface2D
from grid.grid_domain import GridDomain
from common.polynom_factory import local_gradgrad_matrix, local_funcfunc_matrix


class MatrixAssembler2D():

    def __init__(self, assembly_interface: AssemblyInterface2D, grid_domain: GridDomain):
        self.assembly_interface = assembly_interface
        self.grid_domain = grid_domain
        self._gg_matrices = {}
        self._ff_matrices = {}
        self._fg_matrices = {}

        self.dist_dict = {'lobatto': 'globatto', 'uniform': 'uniform'}
        self._matrices_h5_storages = {'lobatto': '/home/lshtanko/Programming/another-fem-framework/datasources/globatto_matrices.h5'}

        self._get_local_matrices_from_file('lobatto')

    def _get_local_ff_matrix(self, distribution, order, cell):
        if self._ff_matrices.get((order, distribution, cell.size)) is None:
            self._ff_matrices[(order, distribution, cell.size)] = local_funcfunc_matrix(dim=2, distribution=self.dist_dict[distribution], order=order)
        return self._ff_matrices[(order, distribution, cell.size)]

    def _get_local_gg_matrix(self, distribution, order):
        if self._gg_matrices.get((order, distribution)) is None:
            self._gg_matrices[(order, distribution)] = local_gradgrad_matrix(dim=2, distribution=self.dist_dict[distribution], order=order)
        return self._gg_matrices[(order, distribution)]

    def _get_local_matrices_from_file(self, distribution):
        M_source = h5py.File(self._matrices_h5_storages[distribution], mode='r')
        orders_available = M_source['order_range'][()]

        size_mass = [(0, 1), (0, 2), (0, 4), (0, 8), (0, 16), (0, 32), (0, 64), (0, 128)]
        size_mass_mapped = [(0., float(x[1]) / float(self.grid_domain.integer_size[0])) for x in size_mass]

        for order in orders_available:
            self._gg_matrices[(order, distribution)] = M_source['grad_matrices/M_order_{}'.format(order)][()]

            for sm, smm in zip(size_mass, size_mass_mapped):
                self._ff_matrices[(order, distribution, (sm[1], sm[1]))] = M_source['mass_matrices/base_M_order_{}'.format(order)][()] * (smm[1] ** 2)

        M_source.close()

    def assemble(self):
        from matplotlib import pyplot as plt
        glob_matrix = csr_matrix((self.assembly_interface.get_ddof_count(), self.assembly_interface.get_ddof_count()))
        for num, props in enumerate(self.assembly_interface.iterate_ddofs_and_wconn()):
            trans_matrix = coo_matrix(([1] * len(props['ddofs']), (list(range(len(props['ddofs']))),props['ddofs'])),
                                      shape=(len(props['ddofs']), self.assembly_interface.get_ddof_count())).tocsr()

            local_gg = self._get_local_gg_matrix(distribution=props['cell_props'][1], order=props['cell_props'][0])
            local_ff = self._get_local_ff_matrix(distribution=props['cell_props'][1], order=props['cell_props'][0], cell=props['cell'])

            glob_matrix += trans_matrix.T * np.dot(np.linalg.inv(local_ff),local_gg) * trans_matrix
            if num % 100 == 0:
                plt.spy(glob_matrix)
                plt.show()
        plt.spy(glob_matrix)
        plt.show()
