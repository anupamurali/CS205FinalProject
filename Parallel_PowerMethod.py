import numpy as np
from mpi4py import MPI
      
if __name__ == '__main__':
  import sys
  import time
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  
  size = comm.Get_size()
  print size
  random_array = np.arange(1000)
  random_fft = np.fft.fft(random_array)
  some_index = 100
  some_frequency = random_fft[some_index]
  n_root_processes = 2
  root_processes = [0,2]
  #non_root_processes = list(set(range(size)) - set(root_processes))
  for i in root_processes:
    if rank != i:
        comm.Send(some_frequency, dest = i)
           
    if rank == i:
        process_G = np.zeros([size/n_root_processes, size], dtype = np.dtype('complex128'))
        other_values = np.empty(size,dtype = np.dtype('complex128'))
        other_values[i] = some_frequency
        for j in range(size):
            if j!=rank:
                other_frequency = np.empty([1,1], dtype = np.dtype('complex128'))
                comm.Recv(other_frequency, source = j)
                other_values[j] = other_frequency[0,0]
            #process_G[rank,j] = some_frequency* np.conjugate( other_frequency[0,0] )
        
        for j in range(size/n_root_processes):
            for k in range(size):
                print j,k
                process_index = i+j #offset by root process
                process_G[j,k] = other_values[process_index]*np.conjugate(other_values[k])
        print rank, process_G

  