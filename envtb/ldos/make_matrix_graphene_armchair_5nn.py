import numpy as np
from scipy import sparse

e0 = -0.034 
g1 = -2.681
g2 = 0.003
g3 = -0.049
g4 = 0.072
g5 = 0.004

a = 1.42
dx = 3 * a

def make_H0(n):
   
   n_half = n / 2
   
   m = np.zeros((n,n), dtype = complex)
   
   m0 = np.diag(e0 * np.ones(n_half, dtype = complex)) +\
        np.diag(g1 * np.ones(n_half-1, dtype = complex), 1) +\
        np.diag(g1 * np.ones(n_half-1, dtype = complex), -1) +\
        np.diag(g2 * np.ones(n_half-2, dtype = complex), 2) +\
        np.diag(g2 * np.ones(n_half-2, dtype = complex), -2) +\
        np.diag(g4 * np.ones(n_half-3, dtype = complex), 3) +\
        np.diag(g4 * np.ones(n_half-3, dtype = complex), -3)
   
   mI = np.diag(g3 * np.ones(n_half, dtype = complex)) +\
        np.diag(g2 * np.ones(n_half-1, dtype = complex), 1) +\
        np.diag(g2 * np.ones(n_half-1, dtype = complex), -1) +\
        np.diag(g5 * np.ones(n_half-3, dtype = complex), 3) +\
        np.diag(g5 * np.ones(n_half-3, dtype = complex), -3) +\
        np.diag(g4 * np.ones(n_half-2, dtype = complex), 2) +\
        np.diag(g4 * np.ones(n_half-2, dtype = complex), -2)
   
   for i in xrange(1, n_half, 2):
      mI[i,i] = g1
      try:
         mI[i, i+2] = g3
      except:
         pass
      try:
         mI[i, i-2] = g3
      except:
         continue
   
   m[:n_half, :n_half] = m0[:,:]
   m[n_half:, n_half:] = m0[:,:]
   m[n_half:, :n_half] = mI[:,:]
   m[:n_half, n_half:] = mI[:,:]
   
   return m

def make_HI(n):
   n_half = n / 2

   m = np.zeros((n,n), dtype = complex)
   
   m1 = np.diag(g5 * np.ones(n_half, dtype = complex))
   m2 = np.diag(g5 * np.ones(n_half, dtype = complex))
   mI = np.diag(g1 * np.ones(n_half, dtype = complex)) +\
        np.diag(g2 * np.ones(n_half-1, dtype = complex), 1) +\
        np.diag(g2 * np.ones(n_half-1, dtype = complex), -1) +\
        np.diag(g5 * np.ones(n_half-3, dtype = complex), 3) +\
        np.diag(g5 * np.ones(n_half-3, dtype = complex), -3) +\
        np.diag(g3 * np.ones(n_half-2, dtype = complex), 2) +\
        np.diag(g3 * np.ones(n_half-2, dtype = complex), -2)


   for i in xrange(1, n_half+1, 2):
      
      try:
         mI[i,i] = g3
      except: pass
         
      try: m1[i, i+1] = g4
      except: pass

      try: m1[i, i-1] = g4
      except: pass
      
      try: m2[i-1, i] = g4
      except: pass
      
      if i > 2:
         try: m2[i-1, i-2] = g4
         except: pass 

      try: mI[i, i+2] = g4
      except: pass
      
      try: mI[i, i-2] = g4
      except: pass
      
   
   m[:n_half, :n_half] = m2[:,:]
   m[n_half:, n_half:] = m1[:,:]
   m[:n_half, n_half:] = mI[:,:]
   
   print m1
   print m2
   print mI

   return m

def block_matrix(m, n):
   
   n_half = n / 2

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
   m[n:, :n] = np.identity(n, dtype = complex)
   
   return m

