import numpy as np
try:
    import matplotlib.pylab as plt
except:
    print 'Warning(plotter): no module matplotlib'
    pass

class Plotter:
    
    def __init__(self, xval=None, yval=None):
        self.xval = xval
        self.yval = yval

    def plotting(self):
        f = plt.figure()
        sp = f.add_subplot(111)
        sp.plot(self.xval, self.yval, 'o-', ms=1.)
        return f
    
    def plot_pcolor(self, X, Y, Z):
        Xm, Ym = np.meshgrid(X, Y)
        try:
            plt.pcolor(Xm, Ym, Z)
        except:
            plt.pcolor(Z)
        plt.colorbar()
        return None 
   
    def plot_density(self, vector, coords, max_el=None, **kwrds):
       
        if max_el == None:
            maxel = np.max(vector)
        else:
            maxel = max_el
                
        xmin = 0.0
        xmax = 0.0
        ymin = 0.0
        ymax = 0.0
        
        for i in xrange(len(vector)):
            
            if maxel != 0.0:
                msize = vector[i] * 150. / np.sqrt(np.sqrt(len(vector))) / maxel
                #R = 1/(maxel) * (density1d[i].real)
                #G = 1/(maxel) * (maxel - density1d[i].real)
                if vector[i] >= maxel:
                    R = 1
                    G = 0
                else:
                    R = 0.999 / (maxel) * (vector[i].real)
                    G = 0.999 / (maxel) * (maxel - vector[i].real)
                   
                B = 0.0
            else: 
                msize = 0.0
                R = 0.0
                G = 0.0
                B = 0.0
                    
            plt.plot(coords[i][0], coords[i][1], 'o', mfc='k', ms=2)
            plt.plot(coords[i][0], coords[i][1], 'o', mfc=(R,G,B), ms=msize, **kwrds)
            
            if coords[i][0] < xmin:
                xmin = coords[i][0]
            elif coords[i][0] > xmax:
                xmax = coords[i][0]
            elif coords[i][1] < ymin:
                ymin = coords[i][0]
            elif coords[i][1] > ymax:
                ymax = coords[i][1]
        
        dx = (xmax-xmin) / 10.
        dy = (ymax-ymin) / 10.
            
        plt.xlim(xmin - dx, xmax + dx)
        plt.ylim(ymin - dy, ymax + dy)
        
        plt.xlabel(r'$x$', fontsize = 24)
        plt.ylabel(r'$y$', fontsize = 24)
        #plt.axes().set_aspect('equal')
        return None    
    
    def plot_potential(self, ham_mit_pot, ham_bare=None, maxel=None, minel=None, plot_real=True, **kwrds):
        
        xmin = 0.0
        xmax = 0.0
        ymin = 0.0
        ymax = 0.0
        
        if ham_bare is None:
            m = np.array(ham_mit_pot.mtot.diagonal())
        else:
            m = np.array(ham_mit_pot.mtot.diagonal() - ham_bare.mtot.diagonal())
        
        if plot_real:
            m = m.real
        else:
            m = m.imag
        
        if minel is None:
            minel = np.min(m)
        if maxel is None:
            maxel = np.max(m)
        
        print np.min(m), np.max(m)
                              
        for i in xrange(len(m)):
                        
            #if maxel != 0.0:
            msize = 500. / np.sqrt(len(m))
            if  m[i].real >= maxel:
                G = 1
                B = 0
            elif m[i].real <= minel:
                G = 0
                B = 1
            else:
                G = 0.999 / (maxel - minel) * (m[i] - minel)
                B = 0.999 / (maxel - minel) * (maxel - m[i])
                                        
            R = 0.0
            #else: 
            #    msize = 0.0
            #    R = 0.0
            #    G = 0.0
            #    B = 0.0
                    
            plt.plot(ham_mit_pot.coords[i][0], ham_mit_pot.coords[i][1], 'o', mfc='k', ms=2)
            plt.plot(ham_mit_pot.coords[i][0], ham_mit_pot.coords[i][1], 'o', mfc=(R,G,B), ms=msize, **kwrds)
            
            if ham_mit_pot.coords[i][0] < xmin:
                xmin = ham_mit_pot.coords[i][0]
            elif ham_mit_pot.coords[i][0] > xmax:
                xmax = ham_mit_pot.coords[i][0]
            elif ham_mit_pot.coords[i][1] < ymin:
                ymin = ham_mit_pot.coords[i][0]
            elif ham_mit_pot.coords[i][1] > ymax:
                ymax = ham_mit_pot.coords[i][1]
        
        dx = (xmax-xmin) / 10.
        dy = (ymax-ymin) / 10.      
        plt.xlim(xmin - dx, xmax + dx)
        plt.ylim(ymin - dy, ymax + dy)
        plt.xlabel(r'$x$', fontsize = 24)
        plt.ylabel(r'$y$', fontsize = 24)
        #plt.axes().set_aspect('equal')
        return None     
        
        
    

class PlotterElectronDensity(Plotter):
    
    def __init__(self, xval=None, yval=None):
        Plotter.__init__(self, xval, yval)
        
    def plot_electron_density_1d(self, nx, Ny, el_den):
        #x = mm.a * np.arange(0, Ny)
        #el_den_1d = np.zeros(Ny, dtype = float)       
        
        #for i in xrange(Ny):
        #    el_den_1d[i] = np.sum(el_den[:, i])
        
        #plt.plot(x, el_den[nx])
        plt.xlabel(r'$y, m$', fontsize = 24)
        plt.ylabel(r'$\rho_{el}$', fontsize = 24)
        return None
            
class PlotterLocalDensityOfStates(Plotter):            
    
    def __init__(self, xval=None, yval=None):
        Plotter.__init__(self, xval, yval) 
    
    def plot_ldos_1d(self, ldos, Ef, N):
        
        X,Y = np.meshgrid(np.linspace(0, N-1, N), Ef)
        plt.pcolor(X,Y, np.array(ldos).real)
        return None
    
      



    
