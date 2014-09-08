import numpy as np
import make_matrix as mm
import make_matrix_graphene as mmg
import make_matrix_graphene_armchair_5nn as mmg_a
import potential
import copy
try:
    import matplotlib.pylab as plt
except:
    print 'Warning(hamiltonian): no module matplotlib'
    pass
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
        ins.m0 = mtot[:self.Ny, :self.Ny]
        ins.mI = mtot[ :self.Ny, self.Ny: 2*self.Ny]
        return ins

    def make_periodic_x(self):
        if self.mtot is None:
            self.build_hamiltonian()
        mtot = self.mtot.copy().tolil()

        mtot[-self.Ny:,:self.Ny] = mtot[:self.Ny, self.Ny:2*self.Ny]
        mtot[:self.Ny, -self.Ny:] = mtot[self.Ny:2*self.Ny, :self.Ny]

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

    def apply_simple_vector_potential(self, A):
        """
              The function applies a vector potential to the hamiltonian parts (H0 and HI)

        conversion_factor = e/h * Angstrem   is a prefactor (for graphene 1.6 * 10**5)

        A: a vector potential of the form [Ax, Ay]
        """
        conversion_factor = 1.602176487 / 1.0545717*1e5

        nonzero_elements_0 = self.m0.nonzero()
        nonzero_elements_I = self.mI.nonzero()

        phase_matrix_0 = np.exp(1j * conversion_factor * A[0] *
                             np.array([self.coords[nonzero_elements_0[1][k]][0] -
                                       self.coords[nonzero_elements_0[0][k]][0]
                                       for k in xrange(len(nonzero_elements_0[0]))]))*\
                      np.exp(1j * conversion_factor * A[1] *
                             np.array([self.coords[nonzero_elements_0[1][k]][1] -
                                       self.coords[nonzero_elements_0[0][k]][1]
                                       for k in xrange(len(nonzero_elements_0[0]))]))
        phase_matrix_I = np.exp(1j * conversion_factor * A[0] *
                             np.array([self.coords[nonzero_elements_I[1][k]+self.Ny][0] -
                                       self.coords[nonzero_elements_I[0][k]][0]
                                       for k in xrange(len(nonzero_elements_I[0]))]))*\
                      np.exp(1j * conversion_factor * A[1] *
                             np.array([self.coords[nonzero_elements_I[1][k]+self.Ny][1] -
                                       self.coords[nonzero_elements_I[0][k]][1]
                                       for k in xrange(len(nonzero_elements_I[0]))]))
        m_0_data = self.m0.data * phase_matrix_0
        m_I_data = self.mI.data * phase_matrix_I

        m_0 = scipy.sparse.csr_matrix((m_0_data, nonzero_elements_0), shape=(self.Ny, self.Ny))
        m_I = scipy.sparse.csr_matrix((m_I_data, nonzero_elements_I), shape=(self.Ny, self.Ny))

        return self.copy_ins(m0=m_0, mI=m_I)

    def apply_simple_magnetic_field(self, magnetic_B=0, gauge='landau_x'):
        conversion_factor=1.602176487/1.0545717*1e-5  # e/hbar*Angstrem^2

        nonzero_elements_0 = self.m0.nonzero()
        nonzero_elements_I = self.mI.nonzero()
        if gauge == 'landau_x':
             phase_matrix_0 = np.exp(1j * conversion_factor * magnetic_B *
                                   np.array([-0.5 * (self.coords[nonzero_elements_0[1][k]][0] -
                                                      self.coords[nonzero_elements_0[0][k]][0]) *\
                                              (self.coords[nonzero_elements_0[0][k]][1] +
                                                self.coords[nonzero_elements_0[1][k]][1])
                                              for k in xrange(len(nonzero_elements_0[0]))]))
             phase_matrix_I = np.exp(1j * conversion_factor * magnetic_B *
                                   np.array([-0.5 * (self.coords[nonzero_elements_I[1][k] + self.Ny][0] -
                                                      self.coords[nonzero_elements_I[0][k]][0]) *\
                                              (self.coords[nonzero_elements_I[0][k] + self.Ny][1] +
                                                self.coords[nonzero_elements_I[1][k]][1])
                                              for k in xrange(len(nonzero_elements_I[0]))]))
        elif gauge == 'landau_y':
            phase_matrix_0 = np.exp(1j * conversion_factor * magnetic_B *
                                  np.array([0.5 * (self.coords[nonzero_elements_0[1][k]][0] +
                                                    self.coords[nonzero_elements_0[0][k]][0]) *\
                                             (self.coords[nonzero_elements_0[1][k]][1] -
                                               self.coords[nonzero_elements_0[0][k]][1])
                                              for k in xrange(len(nonzero_elements_0[0]))]))
            phase_matrix_I = np.exp(1j * conversion_factor * magnetic_B *
                                  np.array([0.5 * (self.coords[nonzero_elements_I[1][k] + self.Ny][0] +
                                                    self.coords[nonzero_elements_I[0][k]][0]) *\
                                             (self.coords[nonzero_elements_I[1][k] + self.Ny][1] -
                                               self.coords[nonzero_elements_I[0][k]][1])
                                              for k in xrange(len(nonzero_elements_I[0]))]))

        m_0_data = self.m0.data * phase_matrix_0
        m_I_data = self.mI.data * phase_matrix_I

        m_0 = scipy.sparse.csr_matrix((m_0_data, nonzero_elements_0), shape=(self.Ny, self.Ny))
        m_I = scipy.sparse.csr_matrix((m_I_data, nonzero_elements_I), shape=(self.Ny, self.Ny))

        return self.copy_ins(m0=m_0, mI=m_I)

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

    def add_vacancies(self, Nvac=10, vactype='single', sign_variation=True, randseed=1000, E0=10.0):
        Ntotal = self.Nx * self.Ny
        try:
            mt = self.mtot.copy()
        except:
            self.build_hamiltonian()
            mt = self.mtot.copy()

        import random

        random.seed(randseed)
        vacan_position = [random.randrange(0.0, stop=Ntotal, step=1.0) for i in xrange(Nvac)]
        if sign_variation:
            signs = [random.randrange(-1, stop=2, step=2.0) for i in xrange(Nvac)]
        else:
            signs = [1.0 for i in xrange(Nvac)]

        if vactype=='single':
            for i in xrange(Nvac):
                mt[vacan_position[i], vacan_position[i]] += E0 * np.sign(signs[i])
        if vactype=='double':
            for i in xrange(Nvac):
                mt[vacan_position[i], vacan_position[i]] += E0
                pos = random.choice([-1,1])
                mt[vacan_position[i]+pos, vacan_position[i]+pos] -= E0 

        return self.copy_ins_with_new_matrix(mt)

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
        #print dz
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

    def find_lead_solution(self, E=0.0, k=10, sigma=0.0, **kwrds):
        A = scipy.sparse.lil_matrix((2*self.Ny, 2*self.Ny), dtype=complex)
        H_I_ = np.linalg.inv(np.transpose((self.mI).todense()))

        mE = scipy.sparse.dia_matrix((E * np.ones(self.Ny, dtype=complex), np.array([0])), shape=(self.Ny,self.Ny))
        A[:self.Ny, :self.Ny] = scipy.sparse.lil_matrix(np.dot(H_I_, (mE - self.m0).todense()))
        A[:self.Ny, self.Ny:] = scipy.sparse.lil_matrix(np.dot(-H_I_, (self.mI).todense()))
        A[self.Ny:, :self.Ny] = scipy.sparse.dia_matrix((np.ones(self.Ny, dtype=complex), np.array([0])), shape=(self.Ny,self.Ny))

        w, v = linalg.eigs(A.tocsc(), k=k, sigma=sigma, **kwrds)
        return w, v
# end class GeneralHamiltonian


class HamiltonianTB(GeneralHamiltonian):

    def __init__(self, Ny, Nx=1, dx=1.0, dy=1.0):

        GeneralHamiltonian.__init__(self)

        self.m0 = mm.make_H0(Ny)
        self.mI = mm.make_HI(Ny)
        self.Nx = Nx
        self.Ny = Ny
        self.Ntot = Nx * Ny
        self.dx=dx
        self.dy=dy
        self.coords = self.get_position(Nx, Ny, dx=self.dx, dy=self.dy)
        self.mtot = None

    def make_periodic_y(self):
        #mlil = self.mtot.tolil()

        #m0 = mlil[:self.Ny, :self.Ny]
        #mI = mlil[:self.Ny, self.Ny:2 * self.Ny]
        m0 = self.m0.copy()
        m0[0, -1] = -mm.t
        m0[-1, 0] = -mm.t

        return self.copy_ins(m0=m0, mI=mI)

    @staticmethod
    def get_position(Nx, Ny, s=1, dx=0.01, dy=0.01):
        return [[i*dx, j*dx, 0] for i in xrange(Nx) for k in xrange(s)  for j in xrange(Ny)]

#end class HamiltonianTB


class HamiltonianGraphene(GeneralHamiltonian):

    def __init__(self, Ny, Nx=1, rescale=1.0):

        GeneralHamiltonian.__init__(self)
        self.m0 = mmg.make_H0(Ny, rescale=rescale)
        self.mI = mmg.make_HI(Ny, rescale=rescale)
        self.Nx = Nx
        self.Ny = Ny
        self.Ntot = Nx * Ny
        self.coords = self.get_position(Nx, Ny, rescale=rescale)
        self.mtot = None

    @staticmethod
    def get_position(Nx, Ny, s=1, rescale=1.0):

        coords = [HamiltonianGraphene.__calculate_coords_in_slice(j, rescale=rescale) for k in xrange(s) for j in xrange(Ny) ]
        coords += [[coords[j][0] + i * mmg.dx / rescale, coords[j][1]] for i in xrange(1, Nx)  for k in xrange(s) for j in xrange(Ny)]

        return coords

    def make_periodic_y(self):

        mist = np.mod(self.Ny, 4)
        ny = self.Ny
        if mist != 0:
            import sys
            print  'Ny should be devidable by 4! Exiting'
            sys.exit(1)
            ny = self.Ny + 4 - mist

        m0 = mmg.make_periodic_H0(m0=self.m0, n=ny)
        mI = mmg.make_periodic_HI(mI=self.mI, n=ny)

        return self.copy_ins(m0=m0, mI=mI)

    @staticmethod
    def __calculate_coords_in_slice(j, rescale=1.0):
        jn = int(j)/4
        a = mmg.a / rescale
        if np.mod(j,4.) == 1.0:
            return [np.sqrt(3)/2.*a, 3.*a*jn + 1./2.*a]
        elif np.mod(j, 4.) == 2.0:
            return [np.sqrt(3)/2.*a, 3.*a*jn + 3./2.*a]
        elif np.mod(j, 4.) == 3.0:
            return [0, 3.*a*jn + 2.*a]
        elif np.mod(j, 4.) == 0.0:
            return [0, 3*a*jn]

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


    def apply_RashbaSO(self, tR=0.01):
        m0 = self.add_Rashba(m=self.m0.copy(), tR=tR, ham_id=0)
        mI = self.add_Rashba(m=self.mI.copy(), tR=tR, ham_id=1)

        return self.copy_ins(m0=m0, mI=mI)

    def add_Rashba(self, m, tR, ham_id):

        dxl = self.coords[self.Ny][0] - self.coords[0][0]
        d = np.sqrt(self.coords[1][0]**2 + self.coords[1][1]**2)
        pauli_mat = self.pauli_matrices()

        for i in xrange(self.Ny/2):
            for j in xrange(self.Ny/2, self.Ny):
                dx0 = self.coords[j][0] - self.coords[i][0] + dxl * ham_id
                dy0 = self.coords[j][1] - self.coords[i][1]
                ann = np.sqrt(dx0**2 + dy0**2)
                if ann < 1.01*d:
                    m[i,j] += 1j * tR * (pauli_mat[0][0, 1]*dy0 - pauli_mat[1][0, 1]*dx0)
                dx0 = self.coords[i][0] - self.coords[j][0] + dxl * ham_id
                #dy0 = -dy0
                ann = np.sqrt(dx0**2 + dy0**2)
                if ann < 1.01*d:
                    m[j,i] += 1j * tR * (-pauli_mat[0][1, 0]*dy0 - pauli_mat[1][1, 0]*dx0)

        return m

    def apply_DresselhausSO(self, tD = 0.01):
        m0 = self.add_Dresselhaus(m=self.m0.copy(), tD=tD, ham_id=0)
        mI = self.add_Dresselhaus(m=self.mI.copy(), tD=tD, ham_id=1)

        return self.copy_ins(m0=m0, mI=mI)

    def add_Dresselhaus(self, m, tD, ham_id):

        dxl = self.coords[self.Ny][0] - self.coords[0][0]
        d = np.sqrt(self.coords[1][0]**2 + self.coords[1][1]**2)
        pauli_mat = self.pauli_matrices()

        for i in xrange(self.Ny/2):
            for j in xrange(self.Ny/2, self.Ny):
                dx0 = self.coords[j][0] - self.coords[i][0] + dxl * ham_id
                dy0 = self.coords[j][1] - self.coords[i][1]
                ann = np.sqrt(dx0**2 + dy0**2)
                if ann < 1.01*d:
                    m[i,j] += 1j * tD * (pauli_mat[0][0, 1]*dx0 - pauli_mat[1][0, 1]*dy0)
                dx0 = self.coords[i][0] - self.coords[j][0] + dxl * ham_id
                #dy0 = -dy0
                ann = np.sqrt(dx0**2 + dy0**2)
                if ann < 1.01*d:
                    m[j,i] += 1j * tD * (pauli_mat[0][1, 0]*dx0 + pauli_mat[1][1, 0]*dy0)

        return m

    @staticmethod
    def pauli_matrices():
        sigma_x = np.array([[0, 1.0], [1.0, 0]])
        sigma_y = 1j * np.array([[0, -1.0], [1.0, 0]])
        sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
        return [sigma_x, sigma_y, sigma_z]
