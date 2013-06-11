import matplotlib.pyplot as plt
import numpy as np
import ldos.plotter

class WaveFunction():
    
    def __init__(self, vec=None, coords=None):
        self.wf1d = vec
        self.coords = coords
    
    def check_norm(self):
        return np.dot(np.transpose(np.conjugate(self.wf1d)), self.wf1d)
        
    def plot_wave_function(self, maxel=None):
        
        ldos.plotter.Plotter().plot_density(
            np.abs(self.wf1d), self.coords, max_el=maxel) 
        plt.axes().set_aspect('equal')
          
    def find_average_position(self):
        f_aver = sum([np.abs(self.wf1d[i])**2 for i in xrange(len(self.wf1d))])
        x_aver = sum([np.abs(self.wf1d[i])**2 * self.coords[i][0] 
                      for i in xrange(len(self.wf1d))]) / f_aver 
        y_aver = sum([np.abs(self.wf1d[i])**2 * self.coords[i][1] 
                      for i in xrange(len(self.wf1d))]) / f_aver
        
        return x_aver, y_aver        
 
# end class WaveFunction        
 
    
class GaussianWavePacket(WaveFunction):
    
    def __init__(self, coords, ic, p0=[0.0, 0.0], sigma=5.):
        
        self.p0 = p0
        self.ic = ic
        self.sigma = sigma
        self.coords = coords
        self.Ntot = len(coords)
        self.wf1d = self.setup()
    
    def gauss_function(self, i, j):
        
        ic = self.coords[self.ic][0]
        jc = self.coords[self.ic][1]
        
        r = np.sqrt((i - ic)**2 + (j - jc)**2)
        
        res = 1. / self.sigma / np.sqrt(np.pi) * \
              np.exp(- (r**2 / 2./ self.sigma**2)) * \
              complex(np.cos(self.p0[0] * (i-ic) + self.p0[1] * (j-jc)),
                      np.sin(self.p0[0] * (i-ic) + self.p0[1] * (j-jc)))
        return res#np.complex(res, 0.0)
                                       
    def setup(self):
                        
        wp1d = np.array([
            self.gauss_function(self.coords[k][0], self.coords[k][1]) 
            for k in xrange(self.Ntot)])
        
        norm = np.sum(np.dot(np.transpose(np.conjugate(wp1d)), wp1d))
        wp1d = wp1d / np.sqrt(norm) 
        return wp1d 

#end class GaussianWavePacket 


class WaveFunction0(WaveFunction):
    """
    This class creates aninitial wave function from the given hamiltonian,
    i.e. as a sum of all eigenstates up to the fermi energy
    
    ham - hamiltonian
    
    mu - fermi energy
    """
    
    def __init__(self, ham, mu, kT):
        
        self.ham = ham
        self.wf1d = self.setup(mu, kT)
        self.coords = self.ham.coords
         
    def setup(self, mu, kT):
        #wf0 = ham.electron_density(mu, kT)
        w, v = self.ham.eigenvalue_problem()
        wf0 = np.zeros(len(v[:,0]), dtype = complex)
        count = 0
        for i in xrange(len(w)):
            if w[i] <= mu:
                count += 1
                #ldos.plotter.Plotter().plot_density(ham.v[:,i], ham.coords)
                #plt.show()
                wf0[:] += v[:,i]
        print count
        norm = np.sum(np.abs(wf0)**2)
        return wf0 / np.sqrt(norm)  

#end class WaveFunction0    
        