#import h5py

from operator_assembler.assembly_interface import AssemblyInterface2D
from operator_assembler.n_to_1_ddof_allocator import Nto1Allocator2D
from grid.grid_manager import GridManager
from grid.allocator_interface import InterfaceDofAllocator2D
from grid.grid_domain import GridDomain
from operator_assembler.matrix_assembler import MatrixAssembler2D
from scipy.ndimage import imread
import h5py
import numpy as np

from common import visual

data_start = (16, 20)
data_shape = (64, 64)
data_end = tuple([i+j for i,j in zip(data_start, data_shape)])

#h5_data = h5py.File('../../../GM_L3D/600/state/viz_GM_1.h5')['index'][0]
#data = h5_data[data_start[0]:data_end[0], data_start[1]:data_end[1]]
data = imread('/Users/marusy/Programming/model/bhi2_labelled0000.tif')[data_start[0]:data_end[0], data_start[1]:data_end[1]]

grid_domain = GridDomain(integer_size=data_shape, domain_size=(1.,1.))
gm = GridManager()
ifma = InterfaceDofAllocator2D(grid_manager=gm.fit(data=data))

gm.draw_grid()

ma = Nto1Allocator2D(grid_interface=ifma)
ma._make_ddof_index()
grid_domain.make_pointwise_index(ma)

sine_test = grid_domain.vectorize_function(lambda x,y: np.sin(x+y))
ifma2 = AssemblyInterface2D(allocator=ma)
MA = MatrixAssembler2D(assembly_interface=ifma2, grid_domain=grid_domain)
MA.assemble()

product = grid_domain.devectorize_vector(MA.assembled * sine_test)
init = grid_domain.devectorize_function(lambda x,y: np.sin(x+y))

MA.assembled[0]

#rel = {}
#for (kp, vp), (ki, vi) in zip(product.items(), init.items()):
#    rel[kp] = vp/vi

#print(rel)
visual.plot_surface_unstructured_w_dict(
    point_val_dict=grid_domain.devectorize_vector(MA.assembled * sine_test),
    plot_domain_shape=grid_domain.domain_size,
    int_domain_shape=grid_domain.integer_size
)

visual.plot_surface_unstructured_w_dict(
    point_val_dict=grid_domain.devectorize_function(lambda x,y: np.sin(x+y)),
    plot_domain_shape=grid_domain.domain_size,
    int_domain_shape=grid_domain.integer_size
)