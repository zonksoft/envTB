import numpy as np
import scipy.sparse

e0 = -0.126
g1 = -3.145
g2 = -0.042
g3 = -0.35

a = 1.42
dx = np.sqrt(3) * a

def make_H0(n):

    a11 = e0 * np.ones(n, dtype = complex)
    a12 = g1 * np.ones(n, dtype = complex)
    a23 = g2 * np.ones(n, dtype = complex)
    #a34 = g3 * np.ones(n, dtype = complex)
    diags = np.array([0,-1,1,-2,2])
    m =  scipy.sparse.spdiags(np.array([a11, a12, a12, a23, a23]),
                                diags, n, n, format="lil")
    for i in xrange(0, n-2, 2):
        m[i, i+3] = g3
        m[i+3, i] = g3
    return m.tocsr()

def make_HI(n):

    #m = g2 * np.diag(np.ones(n, dtype = complex))
    m = scipy.sparse.dia_matrix((g2 * np.ones(n, dtype=complex), np.array([0])), shape=(n,n))
    m = m.tolil()

    for i in xrange(0, n-3, 4):
        m[i+1, i] = g1
        m[i+2, i+3] = g1
        m[i+2, i+1] = g3
        m[i+1, i+2] = g3
        m[i+1, i+3] = g2
        m[i+2, i] = g2

    for i in xrange(0, n-5, 4):
        m[i+5, i+3] = g2
        m[i+2, i+4] = g2
        m[i+4, i+3] = g3
        m[i+3, i+4] = g3

    return m.tocsr() #sparse.dia_matrix(m)

def make_periodic_H0(m0=None, n=10):

    if m0 is None:
        m0 = make_H0(n)

    m = m0.copy().tolil()

    if n > 4:
        mper = np.array([[g2,g1],[g3,g2]])
        m[:2,-2:] = mper[:,:]
        m[-2:, :2] = np.transpose(mper)[:,:]

    return m

def make_periodic_HI(mI=None, n=10):
    if mI is None:
        mI = make_HI(n)
    m = mI.copy().tolil()
    m[0, -1] = g3
    m[-1, 0] = g3

    return m


def block_matrix(m, n):

    b11 = m[:n/2, :n/2]
    b22 = m[n/2:, n/2:]
    b12 = m[:n/2, n/2:]
    b21 = m[n/2:, :n/2]

    return b11, b22, b12, b21

def make_A(H0_, HI_, E):
    n = len(H0_) 
    m = np.zeros((2*len(H0_), 2*len(H0_)), dtype = complex)

    H_I_ = np.linalg.inv(np.transpose(HI_))

    mE = E * np.identity(n, dtype = complex)

    m[:n, :n] = np.dot(H_I_, mE - H0_)
    m[:n, n:] = np.dot(-H_I_, HI_)
    m[n:, :n] = np.identity(n, dtype = float)

    return m

