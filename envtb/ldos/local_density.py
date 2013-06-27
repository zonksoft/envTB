import numpy as np
import greens_function
import matplotlib.pylab as plt

class LocalDensityOfStates:
    
    def __init__(self, H, bc='closed'):
        self.bc = bc
        self.hamiltonian = H
        self.green = None
    
    def __call__(self, E):
        self.green = greens_function.GreensFunction(self.hamiltonian, E, self.bc)
        
        diags = self.green.get_diagonal_elements()
                
        ldos_line = -2.0 * diags.imag / 2. / np.pi #complex(0.0,1.0) * (diags - np.conjugate(np.transpose(diags))) / 2./ np.pi
        self.LDOS = ldos_line
        
        return ldos_line

class DensityOfStates:
    
    def __init__(self, H, E=np.arange(0, 2, 0.1), bc='closed'):
        self.hamiltonian = H
        self.E = E
        self.bc = bc
        
    def __call__(self, E0):
        local_density = LocalDensityOfStates(H=self.hamiltonian, bc=self.bc)
        return np.sum(local_density(E0))/2./np.pi
    
    def plot_density_of_states(self):
        
        DOS = [self.__call__(E0) for E0 in self.E]
        
        plt.plot(self.E, DOS, 'r')
        plt.xlabel(r'$E$')
        plt.ylabel(r'$DOS$')
        
        return None