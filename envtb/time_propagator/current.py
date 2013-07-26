import envtb.time_propagator.wave_function
import numpy as np
import matplotlib.pylab as plt

class CurrentOperator():
    
    def __init__(self):
        pass
    
    def __call__(self, wf, A=[0.0,0.0]):
        """
        wf - wave function
        A - vector potential of the form [A_x, A_y]  
        the call returns current [j_x, j_y]
        """
        if not isinstance(wf, envtb.time_propagator.wave_function.WaveFunction):
            raise TypeError("wf should be instance of envtb.time_propagator.wave_function.WaveFunction()")
        
        wf_prime_x = []
        wf_prime_y = []
        
        for i in xrange(len(wf.wf1d)-1):
            if abs(wf.coords[i+1][0] - wf.coords[i][0]) > 0:
                wf_prime_x.append((wf.wf1d[i+1]-wf.wf1d[i]) / (wf.coords[i+1][0] - wf.coords[i][0]))
            else:
                wf_prime_x.append(0.0)
            
            if abs(wf.coords[i+1][1] - wf.coords[i][1]) > 0:
                wf_prime_y.append((wf.wf1d[i+1]-wf.wf1d[i]) / (wf.coords[i+1][1] - wf.coords[i][1]))
            else:
                wf_prime_y.append(0.0)
       
        wf_prime_x.append(0)
        wf_prime_y.append(0)
                
        j_x = (np.conjugate(wf.wf1d) * wf_prime_x - 
               wf.wf1d * np.conjugate(wf_prime_x)) * complex(0.0, 1.0)\
                - A[0] * wf.wf1d * np.conjugate(wf.wf1d)
        j_y = (np.conjugate(wf.wf1d) * wf_prime_y -
                wf.wf1d * np.conjugate(wf_prime_y)) * complex(0.0, 1.0)\
                - A[1] * wf.wf1d * np.conjugate(wf.wf1d)
        
        plt.subplot(2,2,1)
        plt.plot(j_x.real)
        plt.subplot(2,2,2)
        plt.plot(j_x.imag)
        
        plt.subplot(2,2,3)
        plt.plot(j_y.real)
        plt.subplot(2,2,4)
        plt.plot(j_y.imag)
        plt.show()
        
        Jx = 0.0
        Jy = 0.0
        
        for i in xrange(len(j_x)-1):
            Jx += j_x[i] * (wf.coords[i+1][0]-wf.coords[i][0])
            Jy += j_y[i] * (wf.coords[i+1][1]-wf.coords[i][1])
        
        return [Jx, Jy]
        