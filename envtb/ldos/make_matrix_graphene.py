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
    a34 = g3 * np.ones(n, dtype = complex)
    diags = np.array([0,-1,1,-2,2,-3,3])
    m =  scipy.sparse.spdiags(np.array([a11, a12, a12, a23, a23, a34, a34]),
                                diags, n, n, format="lil")
    for i in xrange(1, n-3, 2):
        m[i, i+3] = 0.0
        m[i+3, i] = 0.0

    return m

def make_HI(n):
   
    #m = g2 * np.diag(np.ones(n, dtype = complex))
    m = scipy.sparse.dia_matrix((g2 * np.ones(n, dtype=complex), np.array([0])), shape=(n,n))
    m = m.tolil()
    
    for i in xrange(0, n-1, 4):
        m[i+1, i] = g1
    
    for i in xrange(0, n-2, 4):
        m[i+2, i+1] = g3
        m[i+1, i+2] = g3
        m[i+2, i] = g2
    
    for i in xrange(0, n-3, 4):
        m[i+2, i+3] = g1
        m[i+1, i+3] = g2
    
    for i in xrange(0, n-4, 4):
        m[i+2, i+4] = g2
        m[i+4, i+3] = g3
        m[i+3, i+4] = g3
    
    for i in xrange(0, n-5, 4):
        m[i+5, i+3] = g2
        m[i+2, i+4] = g2
        m[i+4, i+3] = g3
        m[i+3, i+4] = g3

    return m #sparse.dia_matrix(m)

def make_periodic_H0(n):
    
    m = (np.diag(e0 * np.ones(n, dtype = complex)) + 
         np.diag(g1 * np.ones(n-1, dtype = complex), 1) + 
         np.diag(g1 * np.ones(n-1, dtype = complex), -1) + 
         np.diag(g2 * np.ones(n-2, dtype = complex), 2) + 
         np.diag(g2 * np.ones(n-2, dtype = complex), -2) +  
         np.diag(g3 * np.ones(n-3, dtype = complex), 3) + 
         np.diag(g3 * np.ones(n-3, dtype = complex), -3))
    
    if n > 4:
        m0 = np.array([[g2,g1],[g3,g2]])
        m[:2,-2:] = m0[:,:]
        m[-2:, :2] = np.transpose(m0)[:,:]
    
    return scipy.sparse.lil_matrix(m)
    
def make_periodic_HI(n):
   
    m = g2 * np.diag(np.ones(n, dtype = complex))

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
    
    m[0, -1] = g3
    m[-1, 0] = g3
    
    return scipy.sparse.lil_matrix(m)


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

