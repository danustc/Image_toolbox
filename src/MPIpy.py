from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()
a_len = size*2
my_a = np.arange(a_len).astype('f4')**rank
if rank ==0:
    root_a = np.empty(shape = (size,a_len), dtype = np.float32)
else:
    root_a = None



comm.Bcast([ndarray, datatype], root =0)

obj = comm.bcast(obj, root = 0)