import hamiltonian
from scipy import sparse
from scipy.sparse import linalg
import numpy as np
import make_matrix as mm

class GreensFunction:
    """
    Class 
    """
    
    def __init__(self, H, E, bc='closed'):
        
        self.bc = bc

        if not isinstance(H, hamiltonian.GeneralHamiltonian):
            raise TypeError("H must be hamiltonian.GeneralHamiltonian, not %s", H.__class__.__name__)

        self.hamiltonian = H.mtot
        
        self.Nx = H.Nx
        self.Ny = H.Ny
        self.Ntot = H.Ntot
        self.E = E
        self.H = H
              
    def __inv_greens_matrix(self, E, H):
        
        zplus = complex(0.0, 1.0) * 10**(-12)
        #sig1 = np.zeros((H.Ny * Nx, H.Ny * Nx), dtype = complex)
        #sig2 = np.zeros((H.Ny * Nx, H.Ny * Nx), dtype = complex)
      
        #if self.bc == 'open':
        #sig1[0,0] = self.calculate_self_energy_for_1d(E, -0.05)
        #sig2[-1,-1] = self.calculate_self_energy_for_1d(E, 0.05)
        matrix = (E + zplus) * sparse.eye(H.Ntot, H.Ntot, k = 0, dtype = complex) - H.mtot
        solver = linalg.factorized(matrix.tocsc())
        return solver
    
    def get_diagonal_elements(self):
        
        vec = np.eye(self.Ntot, self.Ntot)
        Green_solver = self.__inv_greens_matrix(self.E, self.H)
        
        Green_diagonal = np.array([Green_solver(vec[:, i])[i] for i in xrange(self.Ntot)])
        
        return Green_diagonal
    
    def __calculate_self_energy_for_1d(self, Ef, U):
        zplus = complex(0.0,1.0) * 10**(-12)
        ck = (1.-((Ef + zplus - U)/(2. * mm.t)))
        ka = np.arccos(ck)
        sigma = -mm.t * (np.cos(ka) + complex(0.0,1.0) * np.sin(ka))
      
        return sigma


