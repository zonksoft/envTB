import numpy
import scipy.linalg

class LinearInterpolationNOGrid:
    """
    The class takes a function on a uniform, rectangular grid (with orthogonal or 
    non-orthogonal lattice vectors) and calculates the function at any point
    using linear interpolation. It can be a 1D, 2D or 3D grid.
    
    See http://en.wikipedia.org/wiki/Trilinear_interpolation for the math.
    
    Usage::
    
      func=[[0,1],[2,3]]
      start=[0,0]
      latticevecs=[[1e-3,0],[0,1e-3]]
      interp=LinearInterpolationNOGrid(func,latticevecs)
      
      f=interp(0.3,0.6)
    
    Note: the gridpoints on the upper boundary in each direction 
    are not accessible by the interpolation function.
    TODO: fix that.
    
    Also note: Pay attention to the ordering of the ``func`` array.
    
    Also note: Due to the possibility of negative indexing of numpy arrays, copies of
    the data will appear "behind" (=in negative direction) 
    the domain of definition. I kept this feature/bug because it is nice
    for orbitals which sit at the box edge.
    """
    
    #for higher performance, implement bilinear and linear 
    #interpolation, and extra functions for orthogonal grid
    
    __dim=None
    __func=None
    __start=None
    __basisvecs=None
    __transformation_matrix=None
    __default=None
    
    def __init__(self,func,grid=1,start=None,default=None):
        """
        func: multi-dimensional array containing the function
              values (list or numpy.array) in a right-handed coordinate system
              (e.g. x to the left, y up), with the indices in the same order 
              as the grid sizes/lattice vectors (i.e. if the lattice vectors
              point in x,y and z direction, func will be called like 
              func[x,y,z])
        start: the start point of the coordinate system given by latticevecs
               (list or numpy.array). Default is None, which means that
               the grid is starting at the coordinate origin.
        grid: either 1) Lattice vectors (list or numpy.array)
                     or 2) a number, which is the gridsize.
                     Default is 1.
        default: The function value of points outside the area.
                 Default is None.
        """
        
        #Convert lists to numpy.array
        self.__func=numpy.array(func,copy=False)
        self.__dim=len(self.__func.shape)
        self.__default=default
        
        if start==None:
            self.__start=numpy.zeros(self.__dim)
        else:
            self.__start=numpy.array(start,copy=False)  
            
        if isinstance(grid,(int,long,float)): #http://stackoverflow.com/a/3501408/1447622
            basisvecs=grid*numpy.eye(self.__dim)
        else:
            basisvecs=numpy.array(grid,copy=False)
            
        
        self.__basisvecs=basisvecs
            
        self.__transformation_matrix=self.__calc_transformation_matrix(basisvecs)
    
    def __call__(self,*point):
        """
        The interpolation function can be called like::
        
            my_interp(1,2,3)
            
        as well as::
        
            my_interp([1,2,3])
        """
        
        if(len(point)==1):
            point=point[0]
        
        grid_element,relative_coordinates=self.__find_position_in_no_grid(point)
        
        f000=f001=f010=f011=f100=f101=f110=f111=0
        x=y=z=0
        if self.__point_in_domain_of_definition(point):
            if(self.__dim==1):
                f000=self.__func[tuple(grid_element+[0])]
                f100=self.__func[tuple(grid_element+[1])]   
                x,=relative_coordinates         
            if(self.__dim==2):
                f000=self.__func[tuple(grid_element+[0,0])]
                f010=self.__func[tuple(grid_element+[0,1])]
                f100=self.__func[tuple(grid_element+[1,0])]
                f110=self.__func[tuple(grid_element+[1,1])] 
                x,y=relative_coordinates            
            if(self.__dim==3):
                f000=self.__func[tuple(grid_element+[0,0,0])]
                f001=self.__func[tuple(grid_element+[0,0,1])]
                f010=self.__func[tuple(grid_element+[0,1,0])]
                f011=self.__func[tuple(grid_element+[0,1,1])]
                f100=self.__func[tuple(grid_element+[1,0,0])]
                f101=self.__func[tuple(grid_element+[1,0,1])]
                f110=self.__func[tuple(grid_element+[1,1,0])]
                f111=self.__func[tuple(grid_element+[1,1,1])]
                x,y,z=relative_coordinates
                
            return self.__trilinear_interpolation(f000, f001, f010, f011, 
                                                  f100, f101, f110, f111, 
                                                  x, y, z) 
        else:
            return self.__default

    def domain_of_definition_box(self):
        """
        Box that includes the domain of definition (which itself is a
        parallelepiped) in the form (corner1,corner2).
        """

        corner1=self.__start
        corner2=corner1+numpy.dot(self.__func.shape,self.__basisvecs)

        return corner1,corner2
        
    def __point_in_domain_of_definition(self,point):
        """
        Check if a point is within the domain of definition.

        Note: the gridpoints on the upper boundary in each direction 
        are not accessible by the interpolation function.
        TODO: fix that.
        """
        shape = numpy.shape(self.__func)
        grid_element, _ = self.__find_position_in_no_grid(point)
        
        for shape_coord, point_coord in zip(shape, grid_element):
            if point_coord < 0 or point_coord >= shape_coord-1:
                return False
        return True
        
    def filter_points_in_domain_of_definition(self, points):
        """
        Select points from a given list which are within the
        domain of definition.
        """
        return [point for point in points 
                if self.__point_in_domain_of_definition(point) is True]
        
    def __find_position_in_no_grid(self,point):
        """
        Finds the grid element the point is in and its position in this
        element in terms of the NO basis.
        
        point: The point (numpy.array)
        """
        
        point_in_no_basis=numpy.dot(self.__transformation_matrix,
                                    (point-self.__start))
        
        grid_element=numpy.floor(point_in_no_basis)
        relative_coordinates=point_in_no_basis-grid_element
        
        return grid_element,relative_coordinates
            
        
    def __calc_transformation_matrix(self,basisvecs):
        """
        Calculate transformation matrix to find corresponding vector in the 
        NO basis for a vector in cartesian coordinates.
        """
        
        return scipy.linalg.inv(numpy.transpose(basisvecs))
        
    def __trilinear_interpolation(self,f000, f001, f010, f011, f100, f101, 
                                   f110, f111, x, y, z):
        return (f111*x*y*z +
                f110*x*y*(1 - z) +
                f101*x*(1 - y)*z +
                f100*x*(1 - y)*(1 - z) +
                f011*(1 - x)*y*z +
                f010*(1 - x)*y*(1 - z) +
                f001*(1 - x)*(1 - y)*z +
                f000*(1 - x)*(1 - y)*(1 - z))
        
    def map_function_to_points(self,points):
        """
        Return list of function values at given points.
        
        points: list of points
        """
        return [self.__call__(point) for point in points]
    
    def dim(self):
        """
        Return the dimension of the interpolation function.
        """
        
        return self.__dim

