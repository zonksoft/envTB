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

