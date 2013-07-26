import numpy as np

class Potential1D:
    
    def range(self):
        """
        Gives the range [xmin,xmax] of the potential.
        Returns None for a range which is infinite.
        """
        raise NotImplementedError

    def __call__(self,x):
        """
        Returns the value of the potential at x.
        """
        pass


class Potential1DFromFunction(Potential1D):
    
    def range(self, xmin = None, xmax = None):
        """
        Gives the range [xmin,xmax] of the potential.
        """
        return [xmin, xmax]

    def __call__(self,x):
        """
        Returns the value of the potential at x.
        """
        return self.potential(x)

    def __init__(self, f):
        """
        docstring
        """
        self.potential = f


class Potential2D:
    def range(self):
        """
        Gives the range [xmin,xmax] of the potential.
        Returns None for a range which is infinite.
        """
        pass

    def __call__(self,x):
        """
        Returns the value of the potential at x.
        """
        pass


class Potential2DFromFunction(Potential2D):
    def range(self, xmin = None, xmax = None, ymin = None, ymax = None):
        """
        Gives the range [xmin,xmax] of the potential.
        """
        return [xmin, xmax], [ymin, ymax]

    def __call__(self,r):
        """
        Returns the value of the potential at r.
        r is a list of [x,y]
        """
        return self.potential(r)

    def __init__(self, f):
        """
        docstring
        """
        self.potential = f

#end class Potential2DFromFunction

class SoftConfinmentPotential:
    def __init__(self, da=3., side="All", max_x=100., max_y=100.):
        '''
        The SoftConfinmentPotential gives an exponential decaying function of width da 
        at the boundaries of the flake. One can apply this potential to any boundary by
        specifying side key word.
        
        side - string: "All" - potential is applied to all sides
                       "1,2,4" - list with any numbers from 1 to 4 specifies sides to 
                                 which the potential should be applied. Numeration starts
                                 at x = 0 boundary and goes clockwise through the boundaries.
        
        TODO:
        Note: not applicable for any boundary
        '''
        self.da = da
        if side == "All":
            side = "1,2,3,4"
        self.side = side.split(',')
        self.max_x = max_x
        self.max_y = max_y
        
    
    def __call__(self, r):
        """
        Returns the value of the potential at r.
        r is a list of [x,y]
        i counter in hamiltonian array (corresponds to position on diagonal of ham matrix with coordinate r)
        """
       
        pot_edge = self.__calculate_edge_potential(r)
        pot_corner = self.__calculate_corner_potential(r)
        
        if pot_corner < 1.0:
            
            return (1.0+(-1)*pot_corner)*10.0
        else:
            if pot_edge < 1.0:
                
                return (1.0+(-1)*pot_edge)*10.0
            else:
                
                return (1.0+(-1)*pot_edge)*10.0
    
    def __smooth_function(self, x):
        #return 0.1 * (self.da - x)
        return abs(np.cos((1.0*x + 1.0*self.da) / self.da * np.pi / 2.))
    
    #def __smooth_function_right(self, x):
        #return 0.1 * (self.da - x)
    #    return abs(np.cos((x + self.da) / self.da * np.pi / 2.))
        
    def __calculate_edge_potential(self, r):
        x = r[0]
        y = r[1]
        if x > self.max_x - self.da:
            x = x - self.max_x
            #pot_amp = 0.001 * (self.da - x)
            pot_amp = self.__smooth_function(x)
        elif y > self.max_y - self.da:
            y = y - self.max_y
            #pot_amp = 0.001 * (self.da - y)
            pot_amp = self.__smooth_function(y)
        elif x < self.da:
            #pot_amp = 0.001 * (self.da - x)
            pot_amp = self.__smooth_function(x)
        elif y < self.da:
            #pot_amp = 0.001 * (self.da - y)
            pot_amp = self.__smooth_function(y)
        else:
            pot_amp = 1.0
        
        return pot_amp
    
    def __calculate_corner_potential(self,r):
         
            x = r[0]
            y = r[1]
            if x >= self.max_x - self.da and y >= self.max_y - self.da:
                x = x - self.max_x
                y = y - self.max_y
                #pot_amp = 0.0001 * (self.da - x)
            elif x <= self.da and y >= self.max_y - self.da:
                x = x
                y = y - self.max_y
                #pot_amp = 0.0001 * (self.da - x)
            elif x <= self.da and y <= self.da:
                x = x
                y = y
                #pot_amp = 0.0001 * (self.da - x)
            elif x >= self.max_x-self.da and y <= self.da:
                x = x - self.max_x
                y = y
                #pot_amp = 0.0001 * (self.da - x)
            else:
                return 1.0
            
            pot_amp = self.__smooth_function(x) * self.__smooth_function(y)
            return pot_amp

#end class SoftConfinmentPotential

class SuperLatticePotential:
    
    def __init__(self, Ny, Nx, pot, coords):
        self.coords = coords
        self.pot = pot # 1d array corresponding to coords
        
        self.Nx = Nx
        self.Ny = Ny
        self.xmax = self.coords[Ny * Nx - 1][1]
        self.ymax = self.coords[Ny][1]
    
    def __call__(self, r):
        '''
            r is a list with coords [x, y]
        '''
        a = 1.42
        
        iy_main = int(r[1] / 3. / a * 4)
        
        irest = np.mod(r[1], 3.*a)
        if abs(irest) <= a/2 + 0.00001:
            iy = iy_main + 1
        elif abs(irest) <= 3.*a/2. + 0.00001:
            iy = iy_main + 2
        elif abs(irest) <= 2. * a + 0.00001:
            iy = iy_main + 3
        else:
            iy = iy_main
        
        ix = int(r[0] / np.sqrt(3) /a)
        
        while iy > self.Ny-1:
            iy = iy - self.Ny
        
        while ix > self.Nx-1:
            ix = ix - self.Nx
        
        index = ix * self.Ny + iy
        
        return self.pot[index]
