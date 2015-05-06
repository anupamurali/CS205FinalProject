import numpy as np
from mpi4py import MPI

def tiled_powermethod(comm,Data):
  rank = comm.Get_rank()
  size = comm.Get_size()
  random_array = Data[rank,:]
  random_fft = np.fft.fft(random_array)
  some_index = 802
  some_frequency = random_fft[some_index]
  if rank != 0:
        comm.Send(some_frequency, dest = 0)    
  if rank == 0:
        G_values = np.empty(size,dtype = np.dtype('complex128'))
        G_values[0] = some_frequency
        for j in range(1,size):
                other_frequency = np.empty([1,1], dtype = np.dtype('complex128'))
                comm.Recv(other_frequency, source = j)
                G_values[j] = other_frequency[0,0]
        #initialise the vector v
        v = np.ones(size)+1j
  #now do the power iteration
        n_iterations = 5
        n = 0
        while n< n_iterations:
        #compute mat-vec product in parallel
            v_temp = np.empty_like(v)
            v_past = np.copy(v)
            for i in range(size):
                v_temp[i] = 0+0j
                for j in range(size):
                    v_temp[i]+= G_values[i]*np.conjugate(G_values[j])*v[j]
            
            v = v_temp/np.linalg.norm(v_temp,2)
            if np.linalg.norm((v-v_past),2) < 1e-15:
                break
            n+=1
        print 'number of iterations: ', n                             
        return v
            
class Power_Method:
    def __init__(self, Data):
        self.Transition = []
        self.Data = Data
        self.n = len(Data)
        
    def form_G(self,peak):
        n = len(self.Data)
        G = np.zeros((n,n),dtype = np.complex)
        ffts = np.fft.fft(self.Data)
        for i in range(n):
            for j in range(n):
                Y1 = ffts[i,peak]
                Y2 = ffts[j,peak]
                G[i,j] = Y1*np.conjugate(Y2)
        self.Transition = G
        return G
        
    def power_iteration(self,n_iterations):
        import numpy as np
        v = np.ones(self.n)+1j#*np.random.random(self.n) #initialise the vector
        i = 0
        while (i<n_iterations):
            v_past = np.copy(v)
            v = np.dot(self.Transition, v)
            v = v/np.linalg.norm(v,2)
            if np.linalg.norm((v-v_past),2) < 1e-15:
                break
                
            i+=1
        print 'no. serial iterations: ', i
        return v
    def Rayleigh_Quotient(self, v):#returns an estimate of the eigenvalue given the eigenvector and matrix
        topline = np.dot(v.conj().T, self.Transition)
        topline = np.dot(topline, v)
        bottomline = np.dot(v.conj().T, v)
        eig = topline/bottomline
        return eig
    
    def deflate(self,eig,v):
        subtraction = eig*np.dot(v.conj().T, v)
        new_T = self.Transition-subtraction
        return new_T
                      
if __name__ == '__main__':
  import sys
  import time
  from scipy.io import loadmat
  Camera = loadmat('test08_vert_fixed.mat')
  Data = Camera['dispall'][0:8,:]
  n_sensors = 8
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  if rank == 0:
    #Data = np.random.random_sample((n_sensors,1000))
    print 'starting serial'
    t0 = time.time()
    iterator = Power_Method(Data)
    G = iterator.form_G(802)
    v_serial = iterator.power_iteration(5)
    elapsed_serial = time.time() - t0
    print 'starting parallel'
  t1 = time.time()
  v_par = tiled_powermethod(comm,Data)
  elapsed_parallel = time.time() - t1
  if rank == 0:
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(1)
    ax.plot(np.real(v_serial), c = 'g', linestyle = ':',marker = 'o')
    ax.plot(np.real(v_par), c = 'b')
    print 'parallel time: ', elapsed_parallel
    print 'serial time: ', elapsed_serial
    plt.show()

        

  
