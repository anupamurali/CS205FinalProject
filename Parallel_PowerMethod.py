import numpy as np
from mpi4py import MPI
      
if __name__ == '__main__':
  import sys
  import time
  from scipy.io import loadmat
  Camera = loadmat('test08_vert_fixed.mat')
  Camera_Data = Camera['dispall'][0:8,:]
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  print rank
  size = comm.Get_size()
  random_array = Camera_Data[rank,:]
  random_fft = np.fft.fft(random_array)
  some_index = 802
  some_frequency = random_fft[some_index]
  n_root_processes = 2
  rows_per_process = size/n_root_processes
  root_processes = [0,4]
  #non_root_processes = list(set(range(size)) - set(root_processes))
  for i in root_processes:
    if rank != i:
        comm.Send(some_frequency, dest = i)    
    if rank == i:
        process_G = np.zeros([rows_per_process, size], dtype = np.dtype('complex128'))
        other_values = np.empty(size,dtype = np.dtype('complex128'))
        other_values[i] = some_frequency
        for j in range(size):
            if j!=rank:
                other_frequency = np.empty([1,1], dtype = np.dtype('complex128'))
                comm.Recv(other_frequency, source = j)
                other_values[j] = other_frequency[0,0]
            #process_G[rank,j] = some_frequency* np.conjugate( other_frequency[0,0] )
        #assemble the cross power spectral density matrix G
        for j in range(rows_per_process):
            for k in range(size):
                process_index = i+j #offset by root process
                print "indices:", i,process_index
                process_G[j,k] = other_values[process_index]*np.conjugate(other_values[k])
        print rank, process_G
        #initialise the vector v
        v = np.ones(size)+1j
  #now do the power iteration
  n_iterations = 3
  if rank in root_processes:
    for n in range(n_iterations):
        #compute mat-vec product in parallel
        v_for_process = np.dot(process_G,v)
        print np.shape(v_for_process)
        # now 'gather' results at process 0, using Send and Recv
        if rank!=0:
           comm.Send(v_for_process, dest = 0)
           
        else: #for process 0 gather the mat-vet result and normalise
            v[0:rows_per_process] = v_for_process
            for j in root_processes[1:]:
                part_of_v = np.empty_like(v_for_process)
                comm.Recv(part_of_v, source = j)
                print np.shape(v[j: j+ rows_per_process])
                v[j: j+ rows_per_process] = part_of_v
            
            v = v/np.linalg.norm(v,2)   
            # now scatter to the other computational processes
            for j in root_processes[1:]:
                comm.Send(v, dest = j)
        #receive the updated v from process 0
        if rank!=0:
            receive_v = np.empty_like(v)
            comm.Recv(receive_v, source = 0)
            v = receive_v
                        
    if rank == 0:
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1)
        ax.plot(-1*np.real(v))
        print v
        plt.show()

  
