import h5py

from operator_assembler.assembly_interface import AssemblyInterface2D
from operator_assembler.n_to_1_ddof_allocator import Nto1Allocator2D
from grid.grid_manager import GridManager
from grid.allocator_interface import InterfaceDofAllocator2D
from scipy.ndimage import imread
from time import time, strftime, gmtime

stime = time()

data_start = (16, 20)
data_shape = (256, 256)
data_end = tuple([i+j for i,j in zip(data_start, data_shape)])

#h5_data = h5py.File('../../../GM_L3D/600/state/viz_GM_1.h5')['index'][0]
#data = h5_data[data_start[0]:data_end[0], data_start[1]:data_end[1]]
data = imread('/Users/leonshting/Programming/Schlumberger/model/bhi2_labelled0000.tif')[data_start[0]:data_end[0], data_start[1]:data_end[1]]

gm = GridManager()
ifma = InterfaceDofAllocator2D(grid_manager=gm.fit(data=data))

ma = Nto1Allocator2D(grid_interface=ifma)
ma._make_ddof_index()
ifma2 = AssemblyInterface2D(allocator=ma)

for i in ifma2.iterate_ddofs_and_wconn():
    print(i)



