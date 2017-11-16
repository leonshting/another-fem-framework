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
            'lobatto': '/home/lshtanko/Programming/another-fem-framework/datasources/globatto_matrices.h5',
            #'lobatto': '/Users/marusy/Programming/another-fem-framework/datasources/globatto_matrices.h5'
            #'lobatto': '/Users/leonshting/Programming/Schlumberger/fem-framework/datasources/globatto_matrices.h5'
        }

        self._get_local_matrices_from_file('lobatto')

        #self.I_b2s = np.array(
        #    [[0.47913998,0.6452334,-0.21919967,0.02834773,0.39265603,-0.39775608,0.07157862],
        #     [0.03097077,0.37368352,0.51547025,0.09433045,-0.05016152,0.07682102,-0.04111448],
        #     [-0.04111448,0.07682102,-0.05016152,0.09433045,0.51547025,0.37368352,0.03097077],
        #     [0.07157862,-0.39775608,0.39265603,0.02834773,-0.21919967,0.6452334,0.47913998]])

        self.I_s2b = np.array([[ 0.81754162,  0.43473261, -0.5263626 ,  0.5263626 , -0.43473261,  0.18245838],
       [ 0.42887393,  0.56763922,  0.08573385, -0.17818385,  0.17044578, -0.07450393],
       [-0.22878392,  1.10846921,  0.05091138,  0.16763862, -0.17699421,  0.07875892],
       [ 0.08005767, -0.13884084,  1.31121   , -0.449045  ,  0.33237084, -0.13574767],
       [-0.06491343,  0.10570322,  0.50879748,  0.70865252, -0.42450822,  0.16626343],
       [ 0.0625    , -0.194645  ,  0.63215   ,  0.63215   , -0.194645  ,  0.0625    ],
       [ 0.16626343, -0.42450822,  0.70865252,  0.50879748,  0.10570322, -0.06491343],
       [-0.13574767,  0.33237084, -0.449045  ,  1.31121   , -0.13884084,  0.08005767],
       [ 0.07875892, -0.17699421,  0.16763862,  0.05091138,  1.10846921, -0.22878392],
       [-0.07450393,  0.17044578, -0.17818385,  0.08573385,  0.56763922,  0.42887393],
       [ 0.18245838, -0.43473261,  0.5263626 , -0.5263626 ,  0.43473261,  0.81754162]])

        self.I_b2s = np.array([[ 0.40878111,  1.21738669, -0.95206492,  0.33314864, -0.18425171,  0.0625    ,  0.47194171, -0.56491364,  0.32774492, -0.21148669,  0.09121889],
       [ 0.0382833 ,  0.28381791,  0.81252745, -0.10177947,  0.05284786, -0.034285  , -0.21225286,  0.24363947, -0.12974245,  0.08522709, -0.0382833 ],
       [-0.03162156,  0.02924139,  0.02546341,  0.65560948,  0.17354209,  0.075955  ,  0.24167791, -0.22452448,  0.08381159, -0.06077139,  0.03162156],
       [ 0.03162156, -0.06077139,  0.08381159, -0.22452448,  0.24167791,  0.075955  ,  0.17354209,  0.65560948,  0.02546341,  0.02924139, -0.03162156],
       [-0.0382833 ,  0.08522709, -0.12974245,  0.24363947, -0.21225286, -0.034285  ,  0.05284786, -0.10177947,  0.81252745,  0.28381791,  0.0382833 ],
       [ 0.09121889, -0.21148669,  0.32774492, -0.56491364,  0.47194171,  0.0625    , -0.18425171,  0.33314864, -0.95206492,  1.21738669,  0.40878111]])

        #self.I_s2b = np.array(
        #    [[ 0.95827995,  0.30970766, -0.41114484,  0.14315723],
        #     [ 0.25809336,  0.74736704,  0.15364203, -0.15910243],
        #     [-0.08767987,  1.03094049, -0.10032303,  0.15706241],
        #     [ 0.02834773,  0.47165227,  0.47165227,  0.02834773],
        #     [ 0.15706241, -0.10032303,  1.03094049, -0.08767987],
        #     [-0.15910243,  0.15364203,  0.74736704,  0.25809336],
        #     [ 0.14315723, -0.41114484,  0.30970766,  0.95827995]]
        #)

        #self.I_b2s = np.array([[ 0.5  ,  0.75 ,  0.   , -0.25 ,  0.   ],
        #                       [ 0.   ,  0.375,  0.25 ,  0.375,  0.   ],
        #                       [ 0.   , -0.25 ,  0.   ,  0.75 ,  0.5  ]])
        #
        #self.I_s2b = np.array([[ 1.   ,  0.   ,  0.   ],
        #    [ 0.375,  0.75 , -0.125],
        #    [ 0.   ,  1.   ,  0.   ],
        #    [-0.125,  0.75 ,  0.375],
        #    [ 0.   ,  0.   ,  1.   ]])

        #self.I_b2s = np.array([[ 0.23843,  1.08482,  0.43502, -1.54173, -0.01661,  0.75761,  0.28435, -0.32496,  0.08306],
        #                       [ 0.08011,  0.38121,  0.21335,  0.60625,  0.00471, -0.17076, -0.24315,  0.14487, -0.0166 ],
        #                       [-0.02353, -0.12682,  0.42165,  0.15965,  0.13808,  0.15965,  0.42165, -0.12682, -0.02353],
        #                       [-0.0166 ,  0.14487, -0.24315, -0.17076,  0.00471,  0.60625,  0.21335,  0.38121,  0.08011],
        #                       [ 0.08306, -0.32496,  0.28435,  0.75761, -0.01661, -1.54173,  0.43502,  1.08482,  0.23843]])
        #
        #self.I_s2b = np.array(
        #    [[0.47685, 0.87222, -0.33455, -0.18066, 0.16613],
        #    [0.39851, 0.76242, -0.3313, 0.28975, -0.11938],
        #    [0.12235, 0.3267, 0.84331, -0.37233, 0.07997],
        #    [-0.56635, 1.2125, 0.41706, -0.34151, 0.2783],
        #    [-0.01661, 0.02563, 0.98194, 0.02563, -0.01661],
        #    [0.2783, -0.34151, 0.41706, 1.2125, -0.56635],
        #    [0.07997, -0.37233, 0.84331, 0.3267, 0.12235],
        #    [-0.11938, 0.28975, -0.3313, 0.76242, 0.39851],
        #    [0.16613, -0.18066, -0.33455, 0.87222, 0.47685]]
        #)

       # self.I_b2s = np.array([[ 0.58752961,  0.73341258, -0.20163508, -0.125     , -0.18463492,  0.27786242, -0.08752961],
       #[-0.03914037,  0.45152847,  0.44695071,  0.125     ,  0.13029929, -0.15377847,  0.03914037],
       #[ 0.03914037, -0.15377847,  0.13029929,  0.125     ,  0.44695071,  0.45152847, -0.03914037],
       #[-0.08752961,  0.27786242, -0.18463492, -0.125     , -0.20163508,  0.73341258,  0.58752961]])

        #self.I_s2b = np.array([[ 1.17504922, -0.39140656,  0.39140656, -0.17504922],
      # [ 0.29337173,  0.90301192, -0.30752192,  0.11113827],
      # [-0.08065374,  0.8938717 ,  0.2606383 , -0.07385626],
      # [-0.125     ,  0.625     ,  0.625     , -0.125     ],
      # [-0.07385626,  0.2606383 ,  0.8938717 , -0.08065374],
      # [ 0.11113827, -0.30752192,  0.90301192,  0.29337173],
      # [-0.17504922,  0.39140656, -0.39140656,  1.17504922]])


        #self.I_b2s = np.array([[ 0.57288245,  0.59966694,  0.0802658 , -0.27140653,  0.01859134],
        #                       [-0.02286845,  0.4179349 ,  0.2098671 ,  0.4179349 , -0.02286845],
        #                       [ 0.01859134, -0.27140653,  0.0802658 ,  0.59966694,  0.57288245]])
        #
        #self.I_s2b = np.array([[ 1.1457649 , -0.18294757,  0.03718267],
        #                       [ 0.29983347,  0.83586979, -0.13570326],
        #                       [ 0.0802658 ,  0.83946839,  0.0802658 ],
        #                       [-0.13570326,  0.83586979,  0.29983347],
        #                       [ 0.03718267, -0.18294757,  1.1457649 ]])

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

    def _get_local_gg_matrix(self, distribution, order):
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

        self._gg_matrices[(5, distribution)] = np.load('/home/lshtanko/Programming/another-fem-framework/datasources/pipka.npz.npy')
        self._ff_matrices[(5, distribution)] = np.load(
            '/home/lshtanko/Programming/another-fem-framework/datasources/pipka_mass.npz.npy')
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
            glob += dist_1.T * csr_matrix(self._get_local_gg_matrix(distribution=props_1[1], order=props_1[0])) * dist_1
            glob += dist_2.T * csr_matrix(self._get_local_gg_matrix(distribution=props_2[1], order=props_2[0])) * dist_2
            return glob
        elif matrix == 'mass':
            glob += dist_1.T * csr_matrix(self._get_local_ff_matrix(distribution=props_1[1], order=props_1[0], cell=cell_1)) * dist_1
            glob += dist_2.T * csr_matrix(self._get_local_ff_matrix(distribution=props_2[1], order=props_2[0], cell=cell_2)) * dist_2
            return glob

    def _distribute_one_cell(self, cell: Cell2D):
        pairtuple = self.assembly_interface.allocator.get_cell_list_of_ddofs(cell=cell)
        props = self.assembly_interface.allocator.get_cell_props(cell)
        dist_1 = distributed_eye_easy(pairtuples=pairtuple, axis2shape=self.assembly_interface.get_ddof_count()).tocoo()
        glob = dist_1.T * csr_matrix(self._get_local_gg_matrix(distribution=props[1], order=props[0])) * dist_1
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
                local_dist_merge2 = []
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

                    local_dist_merge2.append((
                        self._distribute_eye(tmp_peer) +
                        self._distribute_eye(tmp_host) +
                        self._distribute_interpolant(tmp_peer, tmp_host, self.I_s2b) +
                        self._distribute_interpolant(tmp_host, tmp_peer, self.I_b2s)
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

                whole_dist2 = self._distribute_eye(host_dofs_independent) + \
                             self._distribute_eye(peer_dofs_independent)

                print(peer_dofs_independent, host_dofs_independent, host_dofs_to_merge, peer_dofs_to_merge)
                for dist in local_dist_merge:
                    whole_dist += dist

                for dist in local_dist_merge2:
                    whole_dist2 += dist

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
