import numpy as np
import greens_function

class LocalDensityOfStates:
    
    def __init__(self, H, E = 0.0, bc = 'closed'):
        self.bc = bc
        self.hamiltonian = H
        self.green = None
    
    def __call__(self, E):
        self.green = greens_function.GreensFunction(self.hamiltonian, E, self.bc)
        
        diags = self.green.get_diagonal_elements() 
                
        ldos_line = -2.0 * diags.imag / 2. / np.pi #complex(0.0,1.0) * (diags - np.conjugate(np.transpose(diags))) / 2./ np.pi
        self.LDOS = ldos_line
        
        return ldos_line