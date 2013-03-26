import numpy as np
import matplotlib.pylab as plt
import make_matrix as mm
import make_matrix_graphene as mmg
import potential
import envtb.wannier90.w90hamiltonian as w90hamiltonian
import copy
#from scipy.sparse import linalg
#from scipy import sparse

def FermiFunction(x, mu, kT):
    return 1./(1. + np.exp((x - mu).real/kT))

class GeneralHamiltonian:  
 
    def __init__(self, mtot=None, Nx=None, Ny=None, coords=None):
     
        self.mtot = mtot
        self.Nx = Nx
        self.Ny = Ny
        try:
            self.Ntot = Nx * Ny
        except:
            self.Ntot = None
        self.coords = coords
        self.w = None
        self.v = None
        
    def copy_ins(self, mt):
        ins = copy.copy(self)
        ins.mtot = mt
        return ins
    
    def apply_potential(self, U):
        """
        This function apply potential to the 
        """
        if not isinstance(U, potential.Potential1D):
            if not isinstance(U, potential.Potential2D):
                raise TypeError("f has to be instance of Potential1D or Potential2D")
            
        mt = self.mtot.copy()
        
        if isinstance(U, potential.Potential1D):
            
            mt[:,:] += np.diag([U(self.coords[i][1])
                                for i in xrange(self.Ntot)])
                   
        elif isinstance(U, potential.Potential2D):
           
            mt[:,:] += np.diag([U([self.coords[i][0], self.coords[i][1]])
                                for i in xrange(self.Ntot)])
 
        return self.copy_ins(mt) 
                
    def eigenvalue_problem(self):
                
        w, v = np.linalg.eig(self.mtot)
        
        return w, v 
        
    def electron_density(self, mu, kT):
        
        if self.w == None:
            self.w, self.v = self.eigenvalue_problem()
        
        rho = FermiFunction(self.w, mu, kT)
        rho2 = np.dot(self.v, np.dot(
                      np.diag(rho) * np.identity((self.Ny * self.Nx),dtype = float), 
                      np.conjugate(np.transpose(self.v))))
        
        print np.trace(rho2)
        density = np.diag(rho2)
                    
        return density
        
    def make_periodic_x(self):
        mtot = self.mtot.copy()
        mtot[-self.Ny:,:self.Ny] = mtot[:self.Ny, self.Ny:2*self.Ny]
        mtot[:self.Ny, -self.Ny:] = mtot[:self.Ny, self.Ny:2*self.Ny]        
       
        return self.copy_ins(mtot) 

class HamiltonianTB(GeneralHamiltonian):
   
    def __init__(self, Ny, Nx=1):
        
        GeneralHamiltonian.__init__(self)
        
        m0 = mm.make_H0(Ny)
        mI = mm.make_HI(Ny)
        self.mtot = mm.make_H(m0, mI, Nx)
        self.Nx = Nx
        self.Ny = Ny
        self.Ntot = len(self.mtot)
        self.coords = self.get_position()
       
    def make_periodic_y(self): 
        
        m0 = self.mtot[:self.Ny, :self.Ny]
        mI = self.mtot[:self.Ny, self.Ny:2 * self.Ny]
        
        m0[0, -1] = -mm.t
        m0[-1, 0] = -mm.t
        
        mtot = mm.make_H(m0, mI, self.Nx)
       
        return self.copy_ins(mtot) 
        
    def get_position(self):
        return [[i, j, 0] for i in xrange(self.Nx) for j in xrange(self.Ny)]


class HamiltonianGraphene(GeneralHamiltonian):
    
    def __init__(self, Ny, Nx=1):
        
        GeneralHamiltonian.__init__(self)
        
        m0 = mmg.make_H0(Ny)
        mI = mmg.make_HI(Ny)
        self.mtot = mm.make_H(m0, mI, Nx)
        self.Nx = Nx
        self.Ny = Ny
        self.Ntot = len(self.mtot)
        self.coords = self.get_position()
    
    def _instance_with_new_matrix(self, mtot):
        ins = self.__class__(self.Ny, self.Nx)
        ins.mtot = mtot
        return ins 
     
    def get_position(self):
                    
        coords = [self.__calculate_coords_in_slice(j) for j in xrange(self.Ny)]
        coords += [[coords[j][0] + i * mmg.dx, coords[j][1]] for i in xrange(1, self.Nx) for j in xrange(self.Ny)]
        
        return coords           
    
    def __calculate_coords_in_slice(self, j):
        jn = int(j)/4
        if np.mod(j,4.) == 1.0:
            return [np.sqrt(3)/2.*mmg.a, 3.*mmg.a*jn + 1./2.*mmg.a]
        elif np.mod(j, 4.) == 2.0:
            return [np.sqrt(3)/2.*mmg.a, 3.*mmg.a*jn + 3./2.*mmg.a]
        elif np.mod(j, 4.) == 3.0:
            return [0, 3.*mmg.a*jn + 2.*mmg.a]
        elif np.mod(j, 4.) == 0.0:
            return [0, 3*mmg.a*jn]    
        

class HamiltonianFromW90(GeneralHamiltonian):
    
    def __init__(self, HamW90, Nx):
        
        GeneralHamiltonian.__init__(self)
               
        self.mtot = HamW90.maincell_hamiltonian_matrix().toarray()
        self.coords = HamW90.orbitalpositions()       
        self.Nx = Nx
        self.Ny = len(self.mtot) / Nx
        if np.mod(len(self.mtot), Nx) != 0:
            self.Ny += 1
        self.Ntot = len(self.mtot) 
