import numpy as np
from scipy import sparse

e0 = -0.126 
g1 = -3.145
g2 = -0.042
g3 = -0.35

a = 1.42
dx = np.sqrt(3) * a

def make_H0(n):
   
    m = (np.diag(e0 * np.ones(n, dtype = float)) + np.diag(g1 * np.ones(n-1, dtype = float), 1) + np.diag(g1 * np.ones(n-1, dtype = float), -1) + np.diag(g2 * np.ones(n-2, dtype = float), 2) + np.diag(g2 * np.ones(n-2, dtype = float), -2) +  + np.diag(g3 * np.ones(n-3, dtype = float), 3) + np.diag(g3 * np.ones(n-3, dtype = float), -3))
   
    return m #sparse.dia_matrix(m)

def make_HI(n):
   
    m = g2 * np.diag(np.ones(n, dtype = float))

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

    return m #sparse.dia_matrix(m)


def block_matrix(m, n):
  
    b11 = m[:n/2, :n/2]
    b22 = m[n/2:, n/2:]
    b12 = m[:n/2, n/2:]
    b21 = m[n/2:, :n/2]

    return b11, b22, b12, b21

def make_A(H0_, HI_, E):
    n = len(H0_) 
    m = np.zeros((2*len(H0_), 2*len(H0_)), dtype = float)

    H_I_ = np.linalg.inv(np.transpose(HI_))
   
    mE = E * np.identity(n, dtype = float)

    m[:n, :n] = np.dot(H_I_, mE - H0_)
    m[:n, n:] = np.dot(-H_I_, HI_)
    m[n:, :n] = np.identity(n, dtype = float)
   
    return m

