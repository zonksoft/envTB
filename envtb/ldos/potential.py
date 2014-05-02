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

class Potential2DOnGrid(Potential2D):
    def range(self,  xmin = None, xmax = None, ymin = None, ymax = None):
        """
        Gives the range [xmin,xmax] of the potential.
        """

        return [xmin, xmax], [ymin, ymax]

    def __call__(self,r):
        """
        Returns the value of the potential at r.
        r is a list of [x,y]
        """
        ix = int(r[0]/self.dx)
        iy = int(r[1]/self.dy)
        return self.potential[ix,iy]

    def __init__(self, array, dx=1.0, dy=1.0):
        """
        docstring
        """
        self.potential = array
        self.dx=dx
        self.dy=dy

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
    def __init__(self, da=3., side='All', max_x=100., max_y=100., amplitude=10.0):
        '''
        The SoftConfinmentPotential gives an exponential decaying function of width da 
        at the boundaries of the flake. One can apply this potential to any boundary by
        specifying side key word.
        
        side - string: "All" - potential is applied to all sides
                       "armchir" - for making potential at the armchair edge
                       "zigzag" - for making potential at the zigzag edge
        
        imaginary - bool; whether potential is imaginary or not
        
        TODO:
        Note: not applicable for any boundary
        '''
        self.da = da
        
        if side == 'All':
            self.side = 0
        elif side == 'armchair':
            self.side = 12
        elif side == 'zigzag':
            self.side = 34
        #if side == "All":
        #    side = "1,2,3,4"
        #self.side = side.split(',')
        self.max_x = max_x
        self.max_y = max_y
        self.amplitude = amplitude
        
    
    def __call__(self, r):
        """
        Returns the value of the potential at r.
        r is a list of [x,y]
        i counter in hamiltonian array (corresponds to position on diagonal of ham matrix with coordinate r)
        """
       
        pot_edge = self.__calculate_edge_potential(r)
        
        if self.side == 0:
            pot_corner = self.__calculate_corner_potential(r)
            if pot_corner < 1.0:
                return (1.0 + (-1) * pot_corner) * self.amplitude
            else: return (1.0 + (-1) * pot_edge) * self.amplitude
        elif self.side == 12 or self.side == 34:
            #if pot_edge < 1.0:
            return (1.0 + (-1) * pot_edge) * self.amplitude
            #else:
            #    return (1.0+(-1)*pot_edge)*10.0

    def __smooth_function(self, x):
        return abs(np.cos((x + self.da) / self.da * np.pi / 2.))

    def __calculate_edge_potential(self, r):
        x = r[0]
        y = r[1]

        pot_amp = 1.0

        if x > self.max_x - self.da:
            if self.side == 0 or self.side == 12:
                x = x - self.max_x
                pot_amp = self.__smooth_function(x)
            else:
                if y > self.max_y - self.da:
                    y = y - self.max_y
                    pot_amp = self.__smooth_function(y)
                elif y < self.da:
                    pot_amp = self.__smooth_function(y)
        elif x < self.da:
            if self.side == 0 or self.side == 12:
                pot_amp = self.__smooth_function(x)
            else:
                 if y > self.max_y - self.da:
                    y = y - self.max_y
                    pot_amp = self.__smooth_function(y)
                 elif y < self.da:
                    pot_amp = self.__smooth_function(y)

        elif y > self.max_y - self.da:
            if self.side == 0 or self.side == 34:
                y = y - self.max_y
                pot_amp = self.__smooth_function(y)

        elif y < self.da:
            if self.side == 0 or self.side == 34:
                pot_amp = self.__smooth_function(y)

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
