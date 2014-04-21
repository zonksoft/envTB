import numpy as np
try:
    import matplotlib.pylab as plt
except:
    print 'Warning(lanczos): no module matplotlib'
    pass
import wave_function


class LanczosPropagator():
    """ This class is responsible for creating the Krylov subspace with
     orthonormalized functions and for building Lanczos propagator 

    ham: hamiltonian object of the class HamiltonianTB, HamiltonianGraphene,
    HamiltonianFromW90 (see greenextension.hamiltonian)

    wf0: initial wave function

    NK: size of the Krylov subspace
    """

    def __init__(self, wf, ham, NK=6, dt=1.):

        self.Q = []
        if isinstance(wf, wave_function.WaveFunction):
            self.Q.append(wf.wf1d)
        else:
            self.Q.append(wf)
        self.alpha = []
        self.betta = []
        self.NK = NK
        self.dt = dt
        self.ham = ham
        if self.ham.mtot is None:
            self.ham.build_hamiltonian()
        #self.Nx = ham.Nx
        #self.Ny = ham.Ny
        #self.coords = ham.coords
        self.create_subspace()

    def create_subspace(self):
        """
        The create_subspace(ham) function creates Krylov subspace and finds
        alpha and betta coefficients

        ham: hamiltonian object of the class HamiltonianTB,
        HamiltonianGraphene, HamiltonianFromW90
        (see greenextension.hamiltonian)

        Algorithm:

        q = psi_0 / ||psi_0||
        r = Ham * psi_0
        alpha_0 = q * r

        r = r - alpha_0 * q
        betta_0 = ||r||

        for j in xrange(N) (N - subspace size):
            v = q
            q = r/ betta_{j-1}
            self.Q.append(q)
            alpha_j = q * r
            r = r - alpha_j * q
            betta_j = ||r||
        ...

        The function fills in the arrays self.alpha, self.betta and self.Q
        self.Q - storage of Krylov vectors
        self.alpha[i] = <self.Q[i] * H self.Q[i]>
        self.betta[i-1] = <self.Q[i] * H self.Q[i-1]>

        """
        r = self.__applyHwf()

        self.alpha.append(np.dot(np.conjugate(np.transpose(self.Q[-1])), r))

        r -= self.alpha[0] * self.Q[-1]
        self.betta.append(np.sqrt(np.dot(np.conjugate(np.transpose(r)), r)))

        for i in xrange(1, self.NK):
            v = self.Q[-1]
            self.Q.append(r / self.betta[-1])
            r = self.__applyHwf() - self.betta[-1]*v
            self.alpha.append(np.dot(np.conjugate(np.transpose(self.Q[-1])), r))
            r -= self.alpha[-1] * self.Q[-1]

            self.betta.append(np.sqrt(np.dot(np.conjugate(np.transpose(r)), r)))

        return None

    def __applyHwf(self):
        """
        The applyHwf(ham) applies Hamiltonian to the wave function

        ham: hamiltonian object

        Return:

        Hwf = ham * wf

        """

        Hwf = self.ham.mtot.dot(np.array(self.Q[-1]))#np.dot(self.ham.mtot, np.array(self.Q[-1])) 

        return Hwf

    def __add_subspace(self):
        """
        The add_subspace(ham) function adds one size to the Krylov subspace

        ham - hamiltonian matrix of the system

        Return 
        None

        The function adds one element to the vectors self.alpha, self.betta and one function to the self.Q subspace
        """

        self.NK += 1

        r = self.__applyHwf() - self.betta[-2]*self.Q[-2] - \
            self.alpha[-1]*self.Q[-1]

        v = self.Q[-1]
        self.Q.append(r / self.betta[-1])
        r = self.__applyHwf() - self.betta[-1]*v
        self.alpha.append(np.dot(np.conjugate(np.transpose(self.Q[-1])), r))
        r -= self.alpha[-1] * self.Q[-1]
        self.betta.append(np.sqrt(np.dot(np.conjugate(np.transpose(r)), r)))

        return None

    def __build_hamiltonian_in_lanczos_basis(self):
        """
        The build_hamiltonian_in_lanczos_basis() creates the hamiltonian
        matrix in the lanczos basis
            |alpha_0    betta_0    0    0    ...            |
        HL = |betta_0    alpha_1    betta_1    0    ...      |
            |0    betta_1    alpha_2    betta_1    0    ... |

        Return 
        None

        The function fills in the matrix self.HL
        """
        bt = self.betta[:]

        bt.insert(0,0.0)

        NKSpace = len(self.alpha)

        return  self.alpha * np.diag(np.ones(NKSpace, dtype = float)) + \
                bt[:-1] * np.diag(np.ones(NKSpace-1, dtype = float), 1) + \
                bt[1:] * np.diag(np.ones(NKSpace-1, dtype = float), -1)

    def __build_propagator(self):
        """
        THe built_propagator() creates the
        exp(-i*HL*dt/hbar) = Z * exp(-i*Dn*dt/hbar) * Z^deggar

        NOTE:
        hbar = 0.66 * 10**(-15) eV * s (!!!)
        for graphene Dn is in eV

        Return
        btmp: propagator itself

        THe function fills in self.U

        """

        hbar = 0.66 * 10**(-15) 

        HL = self.__build_hamiltonian_in_lanczos_basis()

        dt = self.dt
        w, v = np.linalg.eig(HL)

        U = np.zeros((len(w),len(w)), dtype = complex)

        for i in xrange(len(w)):
            atmp = dt * w[i] / hbar
            U[:,i] = v[:,i] * complex(np.cos(atmp), -np.sin(atmp))

        btmp = np.dot(U, np.conjugate(np.transpose(v)))

        return btmp

    def propagate(self, num_error=10**(-18), regime='SIL'):

        """
        The propagate() function applies Lanczos propagator
        (from build_propagator()) on the given initial wave function wf0

        res = U[:, 0] * Q: where U is a matrix of propagator
                           and Q contains the Krylov subspace of
                           the given wave function wf0

        num_error: numerical error: ||wf_NK - wf_{NK-1}||**2 < num_error

        regime = 'SIL': short iterative lanczos with changing of the time step
                 'TSC': time-step constant and Lanczos space is changed

        Return
        wf_out: one time step evolution of the wf0
        """

        #if self.U == None:

        if regime is not 'SIL':
            if regime is not 'TSC':
                raise NameError("name %(regime)s is not defined" % vars())

        while 1:

            U = self.__build_propagator()
            wf_krylov = U[:,0]

            dwfk = wf_krylov[self.NK-1]

            """
            Check the num_error:
            dwf = wf_NK - wf_{NK-1} = U[NK-1, 0] * self.Q[self.NK-1]
            conver = abs(dwf)**2 = abs(U[NK-1, 0])**2
            check if conver < num_error
            """
            conver = np.abs(dwfk)**2

            if conver < num_error:

                break

            if regime == 'SIL':
                """
                For 'SIL' regime we adjust the time step
                """
                scale = 0.95 * (num_error / conver)**(1./ self.NK)
                self.dt *= max([0.5, scale])
                #print 'num_error', conver

            elif regime == 'TSC':
                """
                For 'TSC' regime we increase the size of the Krylov space
                """
                self.__add_subspace()
                #print 'num_error', conver

        wfk = np.zeros(len(self.Q[0]), dtype = complex)
        for i in xrange(0, self.NK):
            wfk[:] += wf_krylov[i] * self.Q[i]

        wf_out = wave_function.WaveFunction(wfk)
        wf_out.coords = self.ham.coords

        return wf_out, self.dt, self.NK