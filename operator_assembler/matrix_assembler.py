# coding-utf8
from collections import defaultdict
from typing import List

import h5py
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, linalg, eye
import itertools
from collections import Counter
from sklearn.preprocessing import normalize


from common.helpers import distributed_eye_easy, distributed_eye
from grid.cell import Cell2D
from operator_assembler.assembly_interface import AssemblyInterface2D
from grid.grid_domain import GridDomain
from common.polynom_factory import local_gradgrad_matrix, local_funcfunc_matrix



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

        self.I_s2b = np.load('/home/lshtanko/Programming/another-fem-framework/datasources/3_lr.npy')
        self.I_b2s = np.load('/home/lshtanko/Programming/another-fem-framework/datasources/3_rl.npy')

        self.dist = csr_matrix((
            self.assembly_interface.get_ddof_count(),
            self.assembly_interface.get_ddof_count()))

        self.distT = csr_matrix((
            self.assembly_interface.get_ddof_count(),
            self.assembly_interface.get_ddof_count()))

        self.unmerged = csr_matrix((
            self.assembly_interface.get_ddof_count(),
            self.assembly_interface.get_ddof_count()))

        self.straight_dist = csr_matrix((
            self.assembly_interface.get_ddof_count(),
            self.assembly_interface.get_ddof_count()))

        self.whole_dist = csr_matrix((
            self.assembly_interface.get_ddof_count(),
            self.assembly_interface.get_ddof_count()))

        self.mass_unmerged = csr_matrix((
            self.assembly_interface.get_ddof_count(),
            self.assembly_interface.get_ddof_count()))

    def _get_local_ff_matrix(self, distribution, order, cell, diagonal=True, inverse=False):
        if self._ff_matrices.get((order, distribution, cell.size)) is None:
            self._ff_matrices[(order, distribution, cell.size)] = local_funcfunc_matrix(
                dim=2,
                distribution=self.dist_dict[distribution],
                order=order,
            )
        if diagonal:
            if inverse:
                return np.diag(1/(self._ff_matrices[(order, distribution, cell.size)].sum(axis=1)))
            else:
                return np.diag((self._ff_matrices[(order, distribution, cell.size)].sum(axis=1)))
        else:
            if inverse:
                return np.linalg.inv(self._ff_matrices[(order, distribution, cell.size)])
            else:
                return self._ff_matrices[(order, distribution, cell.size)]

    def _get_local_gg_matrix(self, distribution, order, cell: Cell2D, normed_by=False):
        #TODO: DANGER in zero_index
        if self._gg_matrices.get((order, distribution)) is None:
            self._gg_matrices[(order, distribution)] = local_gradgrad_matrix(dim=2,
                                                                             distribution=self.dist_dict[distribution],
                                                                             order=order)[0]
        if normed_by:
            return np.dot(np.linalg.inv(self._get_local_ff_matrix(
                distribution=distribution,
                cell=cell,
                diagonal=True,
                order=order
            )),self._gg_matrices[(order, distribution)])
        else:
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
                    M_source['mass_matrices/base_M_order_{}'.format(order)][()] * (sm[1] ** 2)

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

    def _distribute_one_cell(self, cell: Cell2D, normed=False):
        pairtuple = self.assembly_interface.allocator.get_cell_list_of_ddofs(cell=cell)
        props = self.assembly_interface.allocator.get_cell_props(cell)
        dist_1 = distributed_eye_easy(pairtuples=pairtuple, axis2shape=self.assembly_interface.get_ddof_count()).tocoo()
        glob = dist_1.T * csr_matrix(
            self._get_local_gg_matrix(distribution=props[1], order=props[0], cell=cell, normed_by=normed)) * dist_1
        return glob.tocsr()

    def _distribute_mass_one_cell(self, cell: Cell2D, inverse=False):
        pairtuple = self.assembly_interface.allocator.get_cell_list_of_ddofs(cell=cell)
        props = self.assembly_interface.allocator.get_cell_props(cell)
        dist_1 = distributed_eye_easy(pairtuples=pairtuple, axis2shape=self.assembly_interface.get_ddof_count()).tocoo()
        glob = dist_1.T * csr_matrix(self._get_local_ff_matrix(distribution=props[1],
                                                               order=props[0], cell=cell, inverse=inverse)) * dist_1
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

    def exclude_redundant(self, M):
        mapping = self.assembly_interface.allocator.new_dof_mapping
        len_after = len(set(mapping))
        ind1 = np.arange(self.assembly_interface.get_ddof_count(), dtype=np.int64)
        ind2 = mapping
        vals = np.ones(shape=len(mapping))
        ret_mat = coo_matrix((vals, (ind1, ind2)), shape=(self.assembly_interface.get_ddof_count(), len_after)).tocsr()
        return ret_mat.T * M * ret_mat


    @staticmethod
    def filter_collaboratively(list1: List, list2: List, list1_index=None, list2_index=None):
        def _get_value(elem, ind):
            if ind is None:
                return elem
            else:
                return elem[ind]

        list1_vals = [_get_value(elem, list1_index) for elem in list1]
        list2_vals = [_get_value(elem, list2_index) for elem in list2]

        list1_vals_cp = list(list1_vals)

        for element1 in list1_vals_cp:
            if element1 in list2_vals:

                list1.remove(list1[list1_vals.index(element1)])
                list2.remove(list2[list2_vals.index(element1)])

                list1_vals.remove(element1)
                list2_vals.remove(element1)
        return list1, list2

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

    def csr_matrix_from_dict(self, d, f):
        vals = [f(v) for v in d.values()]
        ind0 = [k[0] for k in d.keys()]
        ind1 = [k[1] for k in d.keys()]
        cMat = coo_matrix((vals, (ind0, ind1)), shape=(self.assembly_interface.get_ddof_count(), self.assembly_interface.get_ddof_count()))
        return cMat.tocsr()

    def normalize_m_by_m(self, matrix1, matrix2):
        return matrix1.multiply(matrix2)

    def normalize_m_by_dict(self, matrix, d, renorm=True):
        tmp = self.normalize_m_by_m(matrix, self.csr_matrix_from_dict(d, lambda x: 1/x))
        return tmp

    def assemble_dist(self, verbose=True):

        outer_contribs_to_vertex_dofs = defaultdict(list)

        for num, props in enumerate(self.assembly_interface.iterate_ddofs_and_wconn()):

            for (k_edge, adj_cells), (w_edge, weak_conns) in zip(props['adj_cells'].items(), props['wconn'].items()):

                tmp_peer = [self.assembly_interface.allocator.new_dof_mapping[i] for i in self.stack_wcon_dofs(weak_conns)]
                tmp_host = self.assembly_interface.allocator.get_flat_list_of_ddofs_global(
                    edge=k_edge, cell=props['cell'])

                vertex_peer_dofs = [tmp_peer[0], tmp_peer[-1]]
                vertex_host_dofs = [tmp_host[0], tmp_host[-1]]

                # doing the same thing for peer-host interpolation and for host-peer but these connections has a bit different data sturcture
                # TODO: probably condense to nested loops
                for num, vertex_dofs in enumerate([vertex_peer_dofs, vertex_host_dofs]):
                    for d in vertex_dofs:
                        # qucik_fix
                        # lets correct this in some way: wconn generator yields index'ly outdated dof_ids

                        cells = [props['cell']] if num == 0 else [i[1] for i in adj_cells]

                        vertex = self.assembly_interface.get_vertex_by_dof_id(d)
                        possibly_contributing = self.assembly_interface.get_ll_vertices_by_vertex(vertex)
                        adj_cells_ll_vertices = set(i.ll_vertex for i in cells)
                        possibly_contributing_ll_vertices = set(i[0] for i in possibly_contributing)

                        contributing_ll_vertex = possibly_contributing_ll_vertices.intersection(adj_cells_ll_vertices)
                        outer_contribs_to_vertex_dofs[d].extend(contributing_ll_vertex)

                        #possibly do some separation of which dofs contributed from what cell

                self.dist += self._distribute_interpolant(tmp_peer, tmp_host, self.I_s2b) + \
                             self._distribute_interpolant(tmp_host, tmp_peer, self.I_b2s)

               # self.distT += self._distribute_interpolant(tmp_peer, tmp_host, self.I_b2s.T) + \
               #              self._distribute_interpolant(tmp_host, tmp_peer, self.I_s2b.T)


        for dof, contributing_ll_vertices in outer_contribs_to_vertex_dofs.items():
            focused_vertex = self.assembly_interface.get_vertex_by_dof_id(dof_id=dof)
            ground_truth_filtered = [i for i in self.assembly_interface.get_ll_vertices_by_vertex(focused_vertex)\
                                     if i[1] != dof]
            gt_cf, c_ll_v_cf = self.filter_collaboratively(
                ground_truth_filtered,
                contributing_ll_vertices,
                list1_index=0,
                list2_index=None
            )
            print(dof, focused_vertex, c_ll_v_cf, gt_cf)

            if len(c_ll_v_cf) == 0 and len(gt_cf) != 0:
                #now we asssume there is only one dof that didnt commit to neigbor
                #self.dist[dof, gt_cf[0][1]] += 1.
                # self.dist[dof] *= 1.5
                pass

            elif len(c_ll_v_cf) != 0 and len(gt_cf) == 0:
                ## very very bad way
                #self.dist[dof] = csr_matrix(self.dist[dof] / self.dist[dof].sum(axis=1))
                pass
            self.dist[dof] = csr_matrix(self.dist[dof] / self.dist[dof].sum(axis=1))


    def assemble_straight_action(self, verbose=True):
        #for layer_num, cell in \
        #   self.assembly_interface.allocator.grid_interface.iterate_cells_fbts(yield_layer_num=True):
        #    self.straight_dist += self._distribute_eye(self.assembly_interface._get_ddofs_for_cell(cell))
        #
        ##just experiment
        #self.straight_dist = csr_matrix(self.straight_dist/self.straight_dist.sum(axis=1))
        self.straight_dist = eye(self.assembly_interface.get_ddof_count())

    def assemble_glob_local(self, verbose=True, normed=True):
        for layer_num, cell in \
           self.assembly_interface.allocator.grid_interface.iterate_cells_fbts(yield_layer_num=True):
            self.unmerged += self._distribute_one_cell(cell, normed=normed)

    def assemble_mass_glob_local(self, verbose=True, inverse=False):
        for layer_num, cell in \
           self.assembly_interface.allocator.grid_interface.iterate_cells_fbts(yield_layer_num=True):
            self.mass_unmerged += self._distribute_mass_one_cell(cell, inverse=inverse)


    def assemble_whole_dist(self):
        self.assemble_dist()
        self.assemble_straight_action()
        self.whole_dist = (self.dist + self.straight_dist)
        #self.whole_distT = (self.distT + self.straight_dist)
        # bad way of normalization
        self.whole_dist = csr_matrix(self.whole_dist / self.whole_dist.sum(axis=1))


    def assemble(self, verbose=True):
        future_ignore = set()
        glob_matrix = csr_matrix((
            self.assembly_interface.get_ddof_count(),
            self.assembly_interface.get_ddof_count()))

        self.half_glob = csr_matrix((
            self.assembly_interface.get_ddof_count(),
            self.assembly_interface.get_ddof_count()))

        self.unmerged = csr_matrix((
            self.assembly_interface.get_ddof_count(),
            self.assembly_interface.get_ddof_count()))

        self.dist = csr_matrix((
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
                #peer_mass_merged = csr_matrix(
                #    (self.assembly_interface.get_ddof_count(), self.assembly_interface.get_ddof_count()))


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

                self.unmerged += peer_merged + host_local
                nonzeros_dist = self.dist.nonzero()
                nonzeros_wd = whole_dist.nonzero()

                set_nz_dist = set([(i,j) for i,j in zip(*nonzeros_dist)])
                set_nz_wd = set([(i, j) for i, j in zip(*nonzeros_wd)])

                intersecting_rows = set(nonzeros_dist[0]).intersection(nonzeros_wd[0])
                inters = set_nz_dist.intersection(set_nz_wd)


                init_operator = peer_merged + host_local
                #init_mass = peer_mass_merged + mass_local

                self.half_glob += init_operator * whole_dist

                for int_row in intersecting_rows:
                    self.dist[int_row] /=2
                    whole_dist[int_row] /= 2
                self.dist += whole_dist

        for gather in gather_evth:

            glob_matrix += gather[1].T * self.half_glob
                #glob_mass_matrix += whole_dist.T * init_mass * whole_dist

        self.assembled = glob_matrix
        self._ass = self.dist.T * self.unmerged * self.dist
        self._half = self.unmerged * self.dist
       # self.assembled_mass = glob_mass_matrix

        return gather_evth
