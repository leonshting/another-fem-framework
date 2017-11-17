# coding-utf8
import h5py
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import itertools
from collections import Counter

from common.helpers import distributed_eye_easy, distributed_eye
from grid.cell import Cell2D
from operator_assembler.assembly_interface import AssemblyInterface2D
from grid.grid_domain import GridDomain
from common.polynom_factory import local_gradgrad_matrix, local_funcfunc_matrix

from common.visual import plot_sparse_pattern


class MatrixAssembler2D():
    def __init__(self, assembly_interface: AssemblyInterface2D, grid_domain: GridDomain):
        self.assembly_interface = assembly_interface
        self.grid_domain = grid_domain
        self._gg_matrices = {}
        self._ff_matrices = {}

        self.dist_dict = {'lobatto': 'globatto', 'uniform': 'uniform'}
        self._matrices_h5_storages = {
            'lobatto': '/home/lshtanko/Programming/another-fem-framework/datasources/1_10_globatto_integrated.h5',
            #'lobatto': '/Users/marusy/Programming/another-fem-framework/datasources/1_10_globatto_integrated.h5'
            #'lobatto': '/Users/leonshting/Programming/Schlumberger/fem-framework/datasources/1_10_globatto_integrated.h5'
        }

        self._get_local_matrices_from_file('lobatto')

        #self.I_s2b = np.load('/Users/leonshting/Programming/Schlumberger/fem-framework/datasources/3_lr.npy')
        #self.I_b2s = np.load('/Users/leonshting/Programming/Schlumberger/fem-framework/datasources/3_rl.npy')

        self.I_s2b = np.load('/home/lshtanko/Programming/another-fem-framework/datasources/4_lr.npy')
        self.I_b2s = np.load('/home/lshtanko/Programming/another-fem-framework/datasources/4_rl.npy')

    def _get_local_ff_matrix(self, distribution, order, cell, diagonal=True):
        if self._ff_matrices.get((order, distribution, cell.size)) is None:
            self._ff_matrices[(order, distribution, cell.size)] = local_funcfunc_matrix(
                dim=2,
                distribution=self.dist_dict[distribution],
                order=order,
            )
        if diagonal:
            return np.diag(self._ff_matrices[(order, distribution, cell.size)].sum(axis=1))
        else:
            return self._ff_matrices[(order, distribution, (0,cell.size[0]))]

    def _get_local_gg_matrix(self, distribution, order, cell: Cell2D):
        #TODO: DANGER in zero_index
        if self._gg_matrices.get((order, distribution)) is None:
            self._gg_matrices[(order, distribution)] = local_gradgrad_matrix(dim=2,
                                                                             distribution=self.dist_dict[distribution],
                                                                             order=order)[0]
        return self._gg_matrices[(order, distribution)]

    def _get_local_matrices_from_file(self, distribution):
        M_source = h5py.File(self._matrices_h5_storages[distribution], mode='r')
        orders_available = M_source['order_range'][()]

        size_mass = [(0, 1), (0, 2), (0, 4), (0, 8), (0, 16), (0, 32), (0, 64), (0, 128)]
        size_mass_mapped = [(0., float(x[1]) / float(self.grid_domain.integer_size[0])) for x in size_mass]

        for order in orders_available:
            self._gg_matrices[(order, distribution)] = M_source['grad_matrices/M_order_{}'.format(order)][()]

            for sm, smm in zip(size_mass, size_mass_mapped):
                self._ff_matrices[(order, distribution, (sm[1], sm[1]))] = \
                    M_source['mass_matrices/base_M_order_{}'.format(order)][()] * (smm[1] ** 2)

        M_source.close()

    def merge_two_cells(self, cell_1, cell_2, matrix='grad'):
        glob = csr_matrix((self.assembly_interface.get_ddof_count(), self.assembly_interface.get_ddof_count()))
        pairtuple_1 = self.assembly_interface.allocator.get_cell_list_of_ddofs(cell=cell_1)
        props_1 = self.assembly_interface.allocator.get_cell_props(cell_1)
        pairtuple_2 = self.assembly_interface.allocator.get_cell_list_of_ddofs(cell=cell_2)
        props_2 = self.assembly_interface.allocator.get_cell_props(cell_2)
        print(pairtuple_1, pairtuple_2)
        dist_1 = distributed_eye(pairtuples=pairtuple_1,
                                 shape=(len(pairtuple_1), self.assembly_interface.get_ddof_count())).tocoo()
        dist_2 = distributed_eye(pairtuples=pairtuple_2,
                                 shape=(len(pairtuple_2), self.assembly_interface.get_ddof_count())).tocoo()
        if matrix == 'grad':
            glob += dist_1.T * csr_matrix(self._get_local_gg_matrix(distribution=props_1[1], order=props_1[0], cell=cell_1)) * dist_1
            glob += dist_2.T * csr_matrix(self._get_local_gg_matrix(distribution=props_2[1], order=props_2[0], cell=cell_2)) * dist_2
            return glob
        elif matrix == 'mass':
            glob += dist_1.T * csr_matrix(self._get_local_ff_matrix(distribution=props_1[1], order=props_1[0], cell=cell_1)) * dist_1
            glob += dist_2.T * csr_matrix(self._get_local_ff_matrix(distribution=props_2[1], order=props_2[0], cell=cell_2)) * dist_2
            return glob

    def _distribute_one_cell(self, cell: Cell2D):
        pairtuple = self.assembly_interface.allocator.get_cell_list_of_ddofs(cell=cell)
        props = self.assembly_interface.allocator.get_cell_props(cell)
        dist_1 = distributed_eye_easy(pairtuples=pairtuple, axis2shape=self.assembly_interface.get_ddof_count()).tocoo()
        glob = dist_1.T * csr_matrix(self._get_local_gg_matrix(distribution=props[1], order=props[0], cell=cell)) * dist_1
        return glob.tocsr()

    def _distribute_mass_one_cell(self, cell: Cell2D):
        pairtuple = self.assembly_interface.allocator.get_cell_list_of_ddofs(cell=cell)
        props = self.assembly_interface.allocator.get_cell_props(cell)
        dist_1 = distributed_eye_easy(pairtuples=pairtuple, axis2shape=self.assembly_interface.get_ddof_count()).tocoo()
        glob = dist_1.T * csr_matrix(self._get_local_ff_matrix(distribution=props[1],
                                                               order=props[0], cell=cell)) * dist_1
        return glob.tocsr()

    # rewrite with consuming interpolants
    def _distribute_interpolant(self, left_dofs, right_dofs, interp):
        trans_matrix_left = distributed_eye_easy([(num, i) for num, i in enumerate(left_dofs)],
                                                 axis2shape=self.assembly_interface.get_ddof_count())
        trans_matrix_right = distributed_eye_easy([(num, i) for num, i in enumerate(right_dofs)],
                                                  axis2shape=self.assembly_interface.get_ddof_count())

        return trans_matrix_left.T * csr_matrix(interp) * trans_matrix_right

    def _distribute_eye(self, dofs):
        dist = distributed_eye_easy([(num, i) for num, i in enumerate(dofs)],
                                    axis2shape=self.assembly_interface.get_ddof_count())
        return dist.T * csr_matrix(np.eye(len(dofs))) * dist

    @staticmethod
    def _order_edges(weak_conns):
        return sorted(weak_conns, key=lambda x: x[0][1])

    def stack_wcon_dofs(self, weak_conns):
        weak_conns = self._order_edges(weak_conns=weak_conns)
        dofs_list = []
        last = None
        for wc in [i[1] for i in weak_conns]:
            dofs_list.extend([i[1] for i in wc if i[1] != last])
            last = dofs_list[-1]
        return dofs_list

    def assemble(self, verbose=True):
        future_ignore = set()
        glob_matrix = csr_matrix((
            self.assembly_interface.get_ddof_count(),
            self.assembly_interface.get_ddof_count()))

        self.half_glob = csr_matrix((
            self.assembly_interface.get_ddof_count(),
            self.assembly_interface.get_ddof_count()))

        #glob_mass_matrix = csr_matrix((
        #    self.assembly_interface.get_ddof_count(),
        #    self.assembly_interface.get_ddof_count()))

        gather_evth = []
        for num, props in enumerate(self.assembly_interface.iterate_ddofs_and_wconn()):
            #bad way probably refactor
            if props['cell'].ll_vertex not in future_ignore:

                peer_merged = csr_matrix(
                    (self.assembly_interface.get_ddof_count(), self.assembly_interface.get_ddof_count()))
        #        peer_mass_merged = csr_matrix(
        #            (self.assembly_interface.get_ddof_count(), self.assembly_interface.get_ddof_count()))


                host_local = self._distribute_one_cell(props['cell'])
                #mass_local = self._distribute_mass_one_cell(props['cell'])

                # TODO: rewrite to multiple merge
                host_dofs_to_merge = []
                peer_dofs_to_merge = []

                local_dist_merge = []
                host_merge_d_cnt = Counter()

                for (k_edge, adj_cells), (w_edge, weak_conns) in zip(props['adj_cells'].items(), props['wconn'].items()):
                    peer_merged += self.merge_two_cells(cell_1=adj_cells[0][1], cell_2=adj_cells[1][1])
                    #peer_mass_merged += self.merge_two_cells(cell_1=adj_cells[0][1],
                    #                                         cell_2=adj_cells[1][1],
                    #                                         matrix='mass')
                    tmp_peer = self.stack_wcon_dofs(weak_conns)
                    tmp_host = self.assembly_interface.allocator.get_flat_list_of_ddofs_global(
                        edge=k_edge, cell=props['cell'])

                    peer_dofs_to_merge.append(tmp_peer)
                    host_dofs_to_merge.append(tmp_host)

                    local_dist_merge.append((
                        0.5 * self._distribute_eye(tmp_peer) +
                        0.5 * self._distribute_eye(tmp_host) +
                        0.5 * self._distribute_interpolant(tmp_peer, tmp_host, self.I_s2b) +
                        0.5 * self._distribute_interpolant(tmp_host, tmp_peer, self.I_b2s)
                    ))



                host_dofs_to_merge_unique = set([i for i in itertools.chain(*host_dofs_to_merge)])
                peer_dofs_to_merge_unique = set([i for i in itertools.chain(*peer_dofs_to_merge)])

                # in case of multiple merges
                for host_d in itertools.chain(*host_dofs_to_merge):
                    host_merge_d_cnt[host_d] += 1

                adj_cells_entities = [j[1] for j in list(itertools.chain(*[i for i in props['adj_cells'].values()]))]
                future_ignore.update([i.ll_vertex for i in adj_cells_entities])
                host_dofs_independent = set(self.assembly_interface.allocator.
                                            get_cell_list_of_ddofs_global(props['cell'])) - \
                                        host_dofs_to_merge_unique
                peer_dofs_independent = set(list(itertools.chain(
                    *[self.assembly_interface.allocator.get_cell_list_of_ddofs_global(cell)
                      for cell in adj_cells_entities]))) - peer_dofs_to_merge_unique

                whole_dist = self._distribute_eye(host_dofs_independent) + \
                             self._distribute_eye(peer_dofs_independent)

                print(peer_dofs_independent, host_dofs_independent, host_dofs_to_merge, peer_dofs_to_merge)
                for dist in local_dist_merge:
                    whole_dist += dist


                gather_evth.append((local_dist_merge,
                                    whole_dist,
                                    self._distribute_eye(host_dofs_independent),
                                    self._distribute_eye(peer_dofs_independent),
                                    host_local,
                                    peer_merged))

                init_operator = peer_merged + host_local
                #init_mass = peer_mass_merged + mass_local

                self.half_glob += init_operator * whole_dist
                glob_matrix += whole_dist.T * init_operator * whole_dist
                #glob_mass_matrix += whole_dist.T * init_mass * whole_dist

            if verbose:
                print('\r', num, end='')
        self.assembled = glob_matrix
       # self.assembled_mass = glob_mass_matrix

        return gather_evth
