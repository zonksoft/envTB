import numpy as np
import make_matrix as mm
import make_matrix_graphene as mmg
import make_matrix_graphene_armchair_5nn as mmg_a
import potential
import copy
import matplotlib.pylab as plt
import scipy.sparse
from scipy.sparse import linalg
import cmath
#from scipy.sparse import linalg
#from scipy import sparse

def FermiFunction(x, mu, kT):
    return 1./(1. + np.exp((x - mu).real/kT))


class GeneralHamiltonian(object):

    def __init__(self, m0=None, mI=None, Nx=None, Ny=None, coords=None):

        self.Nx = Nx
        self.Ny = Ny
        self.m0 = m0
        self.mI = mI
        self.mtot = None
        try:
            self.Ntot = Nx * Ny
        except:
            self.Ntot = None
        self.coords = coords
        self.w = None
        self.v = None

    def build_hamiltonian(self):
        self.mtot = mm.make_H(self.m0, self.mI, self.Nx)

    def copy_ins(self, m0, mI, Ny = None):
        ins = copy.copy(self)
        ins.m0 = m0
        ins.mI = mI
        ins.mtot = None
        if Ny != None:
            ins.Ny = Ny
        return ins

    def copy_ins_with_new_matrix(self, mtot):
        ins = copy.copy(self)
        ins.mtot = mtot
        return ins

    def make_periodic_x(self):
        mtot = self.mtot.copy().tolil()

        mtot[-self.Ny:,:self.Ny] = mtot[:self.Ny, self.Ny:2*self.Ny]
        mtot[:self.Ny, -self.Ny:] = mtot[:self.Ny, self.Ny:2*self.Ny]

        return self.copy_ins_with_new_matrix(mtot.tocsr())

    def apply_potential(self, U, sign_variation=False):
        """
        This function apply potential to the hamiltonian

        U: potential, an instance of Potential1D or Potential2D type

        The applied potential adds to the diagonal elements of the
        hamiltonian.
        """
        if not isinstance(U, potential.Potential1D):
            if not isinstance(U, potential.Potential2D):
                if not isinstance(U, potential.SoftConfinmentPotential):
                    if not isinstance(U, potential.SuperLatticePotential):
                        raise TypeError('''f must be Potential1D or Potential2D
                                         or SoftConfinmentPotential
                                         or SuperLatticePotential, 
                                         not %s''', U.__class__.__name__)
        if self.mtot is None:
            self.build_hamiltonian()

        mt = self.mtot.copy()

        if isinstance(U, potential.Potential1D):

            mdia = scipy.sparse.dia_matrix((np.array([U(self.coords[i][1])
                                for i in xrange(self.Ntot)]), np.array([0])),
                                           shape=(self.Ntot,self.Ntot))
            mt = mt + mdia.tocsr()

        else:

            mdia = np.array([U([self.coords[i][0], self.coords[i][1]])
                                for i in xrange(self.Ntot)])

            if sign_variation:
                mdia[::2] = -1.0 * mdia[::2]

            mdia = scipy.sparse.dia_matrix((mdia, np.array([0])), shape=(self.Ntot,self.Ntot))

            mt = mt + mdia.tocsr()

        return self.copy_ins_with_new_matrix(mt)

    def apply_vector_potential(self, A):
        """
        The function applies a vector potential to the hamiltonian

        conversion_factor = e/h * Angstrem   is a prefactor (for graphene 1.6 * 10**5)

        A: a vector potential of the form [Ax, Ay]
        """

        if self.mtot is None:
            self.build_hamiltonian()
        #TODO: implement vector potential A(r) position dependent
        conversion_factor = 1.602176487 / 1.0545717*1e5

        nonzero_elements = self.mtot.nonzero()

        phase_matrix = np.exp(1j * conversion_factor * A[0] *
                             np.array([self.coords[nonzero_elements[1][k]][0] -
                                       self.coords[nonzero_elements[0][k]][0]
                                       for k in xrange(len(nonzero_elements[0]))]))*\
                      np.exp(1j * conversion_factor * A[1] *
                             np.array([self.coords[nonzero_elements[1][k]][1] -
                                       self.coords[nonzero_elements[0][k]][1]
                                       for k in xrange(len(nonzero_elements[0]))]))
        m_pot_data = self.mtot.data * phase_matrix
        m_pot = scipy.sparse.csr_matrix((m_pot_data, nonzero_elements), shape=(self.Ntot, self.Ntot))

        return self.copy_ins_with_new_matrix(m_pot)

    def apply_magnetic_field(self, magnetic_B=0, gauge='landau_x'):

        if self.mtot is None:
            self.build_hamiltonian()
        conversion_factor=1.602176487/1.0545717*1e-5  # e/hbar*Angstrem^2

        nonzero_elements = self.mtot.nonzero()
        if gauge == 'landau_x':
             phase_matrix = np.exp(1j * conversion_factor * magnetic_B *
                                   np.array([-0.5 * (self.coords[nonzero_elements[1][k]][0] -
                                                      self.coords[nonzero_elements[0][k]][0]) *\
                                              (self.coords[nonzero_elements[0][k]][1] +
                                                self.coords[nonzero_elements[1][k]][1])
                                              for k in xrange(len(nonzero_elements[0]))]))
        elif gauge == 'landau_y':
            phase_matrix = np.exp(1j * conversion_factor * magnetic_B *
                                  np.array([0.5 * (self.coords[nonzero_elements[1][k]][0] +
                                                    self.coords[nonzero_elements[0][k]][0]) *\
                                             (self.coords[nonzero_elements[1][k]][1] -
                                               self.coords[nonzero_elements[0][k]][1])
                                              for k in xrange(len(nonzero_elements[0]))]))

        m_pot_data = self.mtot.data * phase_matrix
        m_pot = scipy.sparse.csr_matrix((m_pot_data, nonzero_elements), shape=(self.Ntot, self.Ntot))

        return self.copy_ins_with_new_matrix(m_pot)

    def eigenvalue_problem(self, k=20, sigma=0.0, **kwrds):
        if self.mtot is None:
            self.build_hamiltonian()
        w,v = linalg.eigs(self.mtot.tocsc(), k=k, sigma=sigma, **kwrds)
        return w, v

    def sorted_eigenvalue_problem(self, k=20, sigma=0.0, **kwrds):
        if self.mtot is None:
            self.build_hamiltonian()
        w,v = self.eigenvalue_problem(k=k, sigma=sigma, **kwrds)
        wsort, vsort = self.__sort_spec(w=w, v=v, sortv=True)
        return wsort, vsort

    def __sort_spec(self, w, v=None, sortv=False):
        isort = np.argsort(w)
        v = np.array(v)
        wsort = np.sort(w)
        if sortv:
            vsort = np.zeros(v.shape, dtype=complex)

            for i in xrange(len(isort)):
                vsort[:,i] = v[:,isort[i]]

            return wsort, vsort
        else:
            return wsort, v

    def get_spec(self, k0, get_wf=False, num_eigs=200):

        m0 = self.m0#.tocsr()
        mI = self.mI
        mIT = self.mI.conjugate().transpose() #mlil[self.Ny:2*self.Ny,:self.Ny].tocsr()

        #print 'mI', mI
        #print 'mIT', mIT.transpose()

        dz = self.coords[self.Ny][0] - self.coords[0][0]
        bloch_phase = cmath.exp(1j * k0 * dz)
        A = m0+ bloch_phase * mI + mIT / bloch_phase
        n_eigs=num_eigs
        #w, v = np.linalg.eig(A)
        if m0.shape[0] <= n_eigs:
            n_eigs = m0.shape[0]-2

        w,v = linalg.eigs(A, k=n_eigs, sigma=0)
        #w,v = np.linalg.eig(A.todense())

        if get_wf:
            wE, wV = self.__sort_spec(w, v, sortv=True)
            return wE, wV
        else:
            wE = self.__sort_spec(w)[0]
            return wE

    def plot_bandstructure(self, krange = np.linspace(0.0,2.5,100), n_eigs=200, **kwrds):
        w = np.array([self.get_spec(k, num_eigs=n_eigs) for k in krange])

        [plt.plot(krange, w[:,i],  ms=2, **kwrds) for i in xrange(len(w[0,:]))]
        #[plt.axhline(y = np.sign(n) * np.sqrt(2. * 1.6 * 10**(-19) * 
        #                                      1.05 * 10**(-34) * 0.82**2 * 
        #                                      10**12 * 300 * np.abs(n))/1.6*10**(19)) for n in range(-6,7)]
        #plt.ylim(-1.5, 1.5)
        plt.xlabel(r'$k_x$', fontsize=26)
        plt.ylabel(r'$E,eV$', fontsize=26)
        #plt.show()
        return None

    def electron_density(self, mu, kT):

        if self.w == None:
            self.w, self.v = self.eigenvalue_problem()

        rho = FermiFunction(self.w, mu, kT)
        rho2 = np.dot(self.v, np.dot(
                      np.diag(rho) * np.identity((self.Ny * self.Nx),dtype = float), 
                      np.conjugate(np.transpose(self.v))))

        density = np.diag(rho2)

        return density

# end class GeneralHamiltonian


class HamiltonianTB(GeneralHamiltonian):

    def __init__(self, Ny, Nx=1):

        GeneralHamiltonian.__init__(self)

        self.m0 = mm.make_H0(Ny)
        self.mI = mm.make_HI(Ny)
        self.Nx = Nx
        self.Ny = Ny
        self.Ntot = Nx * Ny
        self.coords = self.get_position(Nx, Ny)
        self.mtot = None

    def make_periodic_y(self):
        mlil = self.mtot.tolil()

        m0 = mlil[:self.Ny, :self.Ny]
        mI = mlil[:self.Ny, self.Ny:2 * self.Ny]

        m0[0, -1] = -mm.t
        m0[-1, 0] = -mm.t

        return self.copy_ins(m0=m0, mI=mI)

    @staticmethod
    def get_position(Nx, Ny, s=1):
        return [[i, j, 0] for i in xrange(Nx) for k in xrange(s)  for j in xrange(Ny)]

#end class HamiltonianTB


class HamiltonianGraphene(GeneralHamiltonian):

    def __init__(self, Ny, Nx=1):

        GeneralHamiltonian.__init__(self)
        self.m0 = mmg.make_H0(Ny)
        self.mI = mmg.make_HI(Ny)
        self.Nx = Nx
        self.Ny = Ny
        self.Ntot = Nx * Ny
        self.coords = self.get_position(Nx, Ny)
        self.mtot = None


    @staticmethod
    def get_position(Nx, Ny, s=1):

        coords = [HamiltonianGraphene.__calculate_coords_in_slice(j) for k in xrange(s) for j in xrange(Ny) ]
        coords += [[coords[j][0] + i * mmg.dx, coords[j][1]] for i in xrange(1, Nx)  for k in xrange(s) for j in xrange(Ny)]

        return coords

    def make_periodic_y(self):

        mist = np.mod(self.Ny, 4)
        ny = self.Ny
        if mist != 0:
            ny = self.Ny + 4 - mist

        m0 = mmg.make_periodic_H0(ny)
        mI = mmg.make_periodic_HI(ny)

        return self.copy_ins(m0=m0, mI=mI)

    @staticmethod
    def __calculate_coords_in_slice(j):
        jn = int(j)/4
        if np.mod(j,4.) == 1.0:
            return [np.sqrt(3)/2.*mmg.a, 3.*mmg.a*jn + 1./2.*mmg.a]
        elif np.mod(j, 4.) == 2.0:
            return [np.sqrt(3)/2.*mmg.a, 3.*mmg.a*jn + 3./2.*mmg.a]
        elif np.mod(j, 4.) == 3.0:
            return [0, 3.*mmg.a*jn + 2.*mmg.a]
        elif np.mod(j, 4.) == 0.0:
            return [0, 3*mmg.a*jn]

# end class HamiltonianGraphene

class HamiltonianGrapheneArmchair(GeneralHamiltonian):

    def __init__(self, Ny, Nx=1):

        GeneralHamiltonian.__init__(self)

        self.m0 = mmg_a.make_H0(Ny)
        self.mI = mmg_a.make_HI(Ny)
        self.Nx = Nx
        self.Ny = Ny
        self.Ntot = Nx * Ny
        self.coords = self.get_position()
        self.mtot = None


    def get_position(self, s=1):

        coords = [self.__calculate_coords_in_slice(j) for k in xrange(s)  for j in xrange(self.Ny)]
        coords += [[coords[j][0] + i * mmg_a.dx, coords[j][1]] for i in xrange(1, self.Nx) for k in xrange(s)  for j in xrange(self.Ny)]

        return coords

    '''def make_periodic_y(self):

        mist = np.mod(self.Ny, 4)
        ny = self.Ny
        if mist != 0:
            ny = self.Ny + 4 - mist

        m0 = mmg.make_periodic_H0(ny)
        mI = mmg.make_periodic_HI(ny)

        mtot = mm.make_H(m0, mI, self.Nx)

        return self.copy_ins(mtot, ny)'''

    def __calculate_coords_in_slice(self, j):

        jn = int(j) / 2

        if j < self.Ny/2:
            if np.mod(j, 2.) == 0.0:
                return [0.0, np.sqrt(3)/2. * mmg_a.a * j]
            else:
                return [mmg_a.a / 2., np.sqrt(3)/2. * mmg_a.a * j]
        else:

            if np.mod(j - self.Ny / 2, 2.) == 0.0:
                return [3 * mmg_a.a / 2., np.sqrt(3)/2. * mmg_a.a * (self.Ny - j - 1)]
            else:
                return [2. * mmg_a.a, np.sqrt(3)/2. * mmg_a.a * (self.Ny - j - 1)]

# end class HamiltonianGrapheneArmchair


class HamiltonianFromW90(GeneralHamiltonian):

    def __init__(self, HamW90, Nx):

        GeneralHamiltonian.__init__(self)
        self.mtot = HamW90.maincell_hamiltonian_matrix().tocsr()
        self.coords = HamW90.orbitalpositions()
        self.Nx = Nx
        self.Ny = self.mtot.shape[0] / Nx

        if np.mod(self.mtot.shape[0], Nx) != 0:
            self.Ny += 1
        self.Ntot = self.mtot.shape[0]

# end class HamiltonianFromW90

class HamiltonianWithSpin(GeneralHamiltonian):
    """docstring for HamiltonianWithSpin
        by default the class will create a HamiltonianGraphene (with spin degree) if ham is not provided
    """
    def __init__(self, ham=None, Ny=4, Nx=1):
        GeneralHamiltonian.__init__(self)
        if not ham:
            ham = HamiltonianGraphene(Ny=Ny, Nx=Nx)

        '''@TODO: shall implement coords for the W90'''
        self.coords = ham.get_position(Nx=ham.Nx, Ny=ham.Ny, s=2)

        m0s = ham.m0 #mmg.make_H0(Ny)
        mIs = ham.mI #mmg.make_HI(Ny)
        self.m0 = self.make_m_spin(m0s, ham.Ny)
        self.mI = self.make_m_spin(mIs, ham.Ny)

        self.Nx = ham.Nx
        self.Ny = 2*ham.Ny
        self.Ntot = self.Ny * self.Nx
        self.mtot = None

    def make_m_spin(self, m, N):
        m_new = m.tocsr()
        data = m_new.data
        nonzero_elements = m_new.nonzero()
        ne_0 = np.append(nonzero_elements[0], nonzero_elements[0]+N)
        ne_1 = np.append(nonzero_elements[1], nonzero_elements[1]+N)
        non_zero_new = (ne_0, ne_1)

        data_new = np.append(data[:], data[:])

        return scipy.sparse.csr_matrix((data_new, non_zero_new), shape=(2*N, 2*N))


    def apply_Zeeman(self,  magnetic_B=0.0):

        """
            Ez = g*mu_B * B = 0.12*B[T] meV
        """
        m0 = self.m0
        for i in xrange(self.Ny/2):
            m0[i, i] += 0.00012 * magnetic_B
            m0[self.Ny/2+i, self.Ny/2+i] -= 0.00012 * magnetic_B
        mI = self.mI
        #m_pot = mm.make_H(m0, mI,  self.Nx)

        return self.copy_ins(m0=m0, mI=mI)


    def apply_RashbaSO(self, tR = 0.01):
        m0 = self.m0
        mI = self.mI
        pauli_mat = self.pauli_matrices()

        d = np.sqrt(self.coords[1][0]**2 + self.coords[1][1]**2)

        for i in xrange(self.Ny):
            for j in xrange(self.Ny):
                dx0 = self.coords[j][0] - self.coords[i][0]
                dy0 = self.coords[j][1] - self.coords[i][1]
                dxI = self.coords[j+self.Ny][0] - self.coords[i][0]
                dyI = self.coords[j+self.Ny][1] - self.coords[i][1]
                ann0 = np.sqrt(dx0**2 + dy0**2)
                annI = np.sqrt(dxI**2 + dyI**2)

                if ann0 < 1.01 * d:
                    if i > self.Ny / 2: ispin = 1
                    else: ispin = 0
                    if j > self.Ny/2: jspin = 1
                    else: jspin = 0
                    if ispin == jspin: continue
                    else:
                        m0[i,j] += 1j * tR * (pauli_mat[0][ispin, jspin]*dy0 - pauli_mat[1][ispin, jspin]*dx0)

                if annI < 1.01 * d:
                    if i > self.Ny / 2: ispin = 1
                    else: ispin = 0
                    if j > self.Ny/2: jspin = 1
                    else: jspin = 0
                    if ispin == jspin: continue
                    else:
                        mI[i,j] += 1j * tR * (pauli_mat[0][ispin, jspin]*dyI - pauli_mat[1][ispin, jspin]*dxI)

        #m_tot_new = mm.make_H(m0, mI,  self.Nx)
        return self.copy_ins(m0=m0, mI=mI)

    def apply_DresselhausSO(self, tD = 0.01):
        m0 = self.m0
        mI = self.mI
        pauli_mat = self.pauli_matrices()

        d = np.sqrt(self.coords[1][0]**2 + self.coords[1][1]**2)

        for i in xrange(self.Ny):
            for j in xrange(self.Ny):
                dx0 = self.coords[j][0] - self.coords[i][0]
                dy0 = self.coords[j][1] - self.coords[i][1]
                dxI = self.coords[j+self.Ny][0] - self.coords[i][0]
                dyI = self.coords[j+self.Ny][1] - self.coords[i][1]
                ann0 = np.sqrt(dx0**2 + dy0**2)
                annI = np.sqrt(dxI**2 + dyI**2)

                if ann0 < 1.01 * d:
                    if i > self.Ny / 2: ispin = 1
                    else: ispin = 0
                    if j > self.Ny/2: jspin = 1
                    else: jspin = 0
                    if ispin == jspin: continue
                    else:
                        m0[i,j] += 1j * tD * (pauli_mat[0][ispin, jspin]*dx0 - pauli_mat[1][ispin, jspin]*dy0)

                if annI < 1.01 * d:
                    if i > self.Ny / 2: ispin = 1
                    else: ispin = 0
                    if j > self.Ny/2: jspin = 1
                    else: jspin = 0
                    if ispin == jspin: continue
                    else:
                        mI[i,j] += 1j * tD * (pauli_mat[0][ispin, jspin]*dxI - pauli_mat[1][ispin, jspin]*dyI)

        m_tot_new = mm.make_H(m0, mI,  self.Nx)
        return self.copy_ins(m0=m0, mI=mI)

    @staticmethod
    def pauli_matrices():
        sigma_x = np.array([[0, 1.0], [1.0, 0]])
        sigma_y = 1j * np.array([[0, -1.0], [1.0, 0]])
        sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
        return [sigma_x, sigma_y, sigma_z]
