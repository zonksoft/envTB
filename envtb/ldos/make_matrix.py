import numpy as np
import scipy.sparse
from scipy.sparse import linalg

hbar = 1.055*10**(-34)
m = 0.067 * 9.109 * 10**(-31)
a = 1.0 * 10**(-9)
JtoEV = 1./1.6 * 10**(19) 
t = hbar**2 / 2./ m/ a**2 * JtoEV
print t

def make_H0(Np, Ec = 0):
    a = (4. * t + Ec) * np.ones(Np, dtype = complex)
    b = -t * np.ones(Np, dtype = complex)
    diags = np.array([0,-1,1])
    return scipy.sparse.spdiags(np.array([a,b,b]),diags, Np, Np, format="lil")
    #return ((4 * t + Ec) * np.diag(np.ones(Np, dtype = complex))) -\
    #       (t * np.diag(np.ones(Np-1, dtype = complex), 1)) -\
    #       (t * np.diag(np.ones(Np-1, dtype = complex), -1))


def make_periodic_H0(n, Ec = 0):
    m = make_H0(n, Ec).tolil()
    m[0,-1] = -t
    m[-1,0] = -t

    return  m.tocsr()

def make_HI(n):

    return -t * scipy.sparse.eye(n, n, dtype = complex, format="lil")

def make_H(H0, HI, nx):

    ny = H0.shape[0]
    print ny
    #print H0
    #H = np.zeros((nx*ny,nx*ny), dtype = complex)
    H = scipy.sparse.lil_matrix((nx*ny,nx*ny), dtype=complex)

    HIT = HI.transpose().conjugate()

    for i in xrange(nx):
        j = i * ny

        H[j:j+ny,j:j+ny] = H0[:,:]

        try:
            H[j:j+ny,j+ny:j+2*ny] = HI[:,:]
        except:
            None
        try:
            H[j+ny:j+2*ny,j:j+ny] = HIT[:,:]
        except:
            continue
    
    return H.tocsr()


def block_matrix(m, n):
    b11 = m[:n/2, :n/2]
    b22 = m[n/2:, n/2:]
    b12 = m[:n/2, n/2:]
    b21 = m[n/2:, :n/2]

    return b11, b22, b12, b21

def eigenvalue_problem(H0, HI):
    import matplotlib.pylab as plt
    k = np.arange(-2., 2., 0.01)
    E = []
    dx = a
    for i in xrange(len(k)):
        A = H0 + HI * complex(np.cos(k[i]*dx), np.sin(k[i]*dx)) + np.transpose(HI) * complex(np.cos(k[i]*dx), -np.sin(k[i]*dx)) 
        w, v = np.linalg.eig(A)
    #plt.plot(abs(w))
        E.append(w)

    E = np.array(E)
   
    for i in xrange(len(E[0,:])):
        plt.plot(k, E[:,i].real, 'o', ms=1.)
   
    return None

def make_A(H0_, HI_, E):
    n = H0_.shape[0]
    m = scipy.sparse.lil_matrix((2*n, 2*n), dtype = float)
    
    HIT_ = HI_.transpose()
    
    H_I_ = scipy.sparse.lil_matrix(np.linalg.inv(HIT_.todense()))#linalg.inv(HIT_)#np.linalg.inv(np.transpose(HI_))
   
    mE = E * np.identity(n, dtype = float)

    m[:n, :n] = H_I_.dot(mE - H0_)
    m[:n, n:] = -H_I_.dot(HI_)
    m[n:, :n] = scipy.sparse.eye(n, n, dtype = float)
   
    return m.tocsc()

