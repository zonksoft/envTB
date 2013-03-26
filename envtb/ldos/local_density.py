import numpy as np
import greens_function

class LocalDensityOfStates:
    
    def __init__(self, H, E = 0.0, bc = 'closed'):
        self.bc = bc
        self.hamiltonian = H
        self.green = None
        #self.green = GreensFunction(H, E, bc)
        #self.spectral_function = complex(0.0,1.0) * (self.green.Green - np.conjugate(np.transpose(self.green.Green)))
        
        #self.LDOS = np.diag(self.spectral_function)
  
    def __call__(self, E):
        self.green = greens_function.GreensFunction(self.hamiltonian, E, self.bc)
        
        diags = self.green.get_diagonal_elements() 
                
        ldos_line = complex(0.0,1.0) * (diags - np.conjugate(np.transpose(diags)))
        self.LDOS = ldos_line
        
        return ldos_line