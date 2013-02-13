#Mind coordinate system: first coordinate goes down (=row number), second to the right (=column number)
#Mind usage: creatematrix() checks possible new boundary conditions (i.e. fixed voltage somewhere), createinhomogeneity() does not (=faster).
#Mind grid points: assume that a grid point is in the top left corner of its "grid area"

#TODO: container-plot data function doesnt work for more complicated arrangements
#TODO: in Element Operatoranwendung und matrixelementherstellung trennen
#TODO: periodische selbstverbindung moeglich machen; oder extra option "periodisch" mit spiegelungsoption?
#TODO: index-funktion von Element auch fuern Container machen
#TODO: speicherung des operators spezifisch fuer jedes rechteck macht (noch) keinen sinn
#TODO: verbessere fermi_energy/potential kombination
#TODO: sind echte Ladungen korrekt eingebaut? v.a. bei materialabhaengigem laplace? fkt element.inhomogeneity
#TODO: in self consistency cycle & self consistency solver wird fermi_energy eig als fermi_energy/q benutzt, naemlich als spannung
#TODO: gridhoehe ist in graphene bulk DOS hardcodiert!!
#TODO: variablen, die "charge_operator" heissen, sollten das epsilon_0 schon drin haben
#TODO: Container und PeriodicContainer sollten gemergt und vererbt werden
#TODO: zuerst     qcsolver.refresh_environment_contrib(), dann qcsolver.refresh_basisvecs(), sonst falsch - warum?? war nicht reproduzierbar. fkt besser dokumentieren
#TODO: improve documentation of Laplacian2ndOrderWithMaterials
#TODO: expand return value of vector_to_datamatrix with start vector and
#      latticesize for interpolation  
#TODO; make the solution a class which has all the plot and export
#      functions!!!!!11

from .common import Constants
import scipy.sparse.linalg
import collections
import numpy
#import matplotlib
import pylab


class Element:
    """
    Element describes a single grid point/discretization element in a geometry. 
    It saves the rectangle it belongs to (rect), its position (i,j) within its rectangle.
    It supplies a function matrixelements() which, given an operator, returns the matrix  
    elements of the element to its neighbours and the inhomogeneity of the element.
    """
    #: Row index of Element
    i=0
    #: Column index of Element
    j=0
    rect=0
    __matrixelements=0
    __inhomogeneity=0
    __inhomogeneityelements=0
    #: Electrostatic potential of Element 
    potential=0
    #: Charge of element
    charge=0
    #: Dependence of Fermi energy on charge
    fermi_energy_charge_dependence=0
    #: Dependence of charge on Fermi energy
    charge_fermi_energy_dependence=None
    #: Electrochemical potential of Element
    fermi_energy=0
    #: Dielectric constant of Element
    epsilon=0
    #: Neumann boundary condition of Element
    neumannbc=None
    
    def __init__(self,rect,i,j,potential=None,charge=0,epsilon=1,fermi_energy_charge_dependence=None,fermi_energy=None,neumannbc=None,charge_fermi_energy_dependence=None):
        """
        i: Row index
        j: Column index
        potential: If potential is None, the element is a normal gridpoint.
                        Otherwise, the element has a fixed potential (i.e. a metal/a capacitor).
        charge: charge of the element. Default is 0. If potential!=None, the charge value 
                will be ignored (metal is not charged).
        fermi_energy_charge_dependence: How the fermi energy of the material depends on the charge. Default is None.
                                        Mind that this setting assumes that the element is in a homogeneous environment.
        charge_fermi_energy_dependence: Like the former, but the other way round.
        fermi_energy: If the Fermi energy depends on the number of charge carriers, the fermi energy (=applied voltage e.g. by a battery)
                      can be different from the electrostatic potential. fermi_energy_charge_dependence has to be defined in this case.
                      Then you can calculate the quantum capacitance of the system.
                      If fermi_energy_charge_dependence=None, then fermi_energy=potential.
        epsilon: Relative dielectrical constant.
        neumannbc: The slope along x or y direction can be fixed, e.g. for a neumann boundary condition.
                   E.g. neumannbc=(14,'y') or neumannbc=(0,'x'). neumannbc and potential cannot be used
                   at the same time. Charge has to be 0 (=default value).
                   Values != 0 do not seem to work right (see comments).
                      
        """
        self.i=i
        self.j=j
        self.rect=rect
        self.potential=potential
        self.charge=charge
        self.fermi_energy_charge_dependence=fermi_energy_charge_dependence
        self.charge_fermi_energy_dependence=charge_fermi_energy_dependence
        self.fermi_energy=fermi_energy
        if fermi_energy==None:
            fermi_energy=potential
        self.epsilon=epsilon
        self.neumannbc=neumannbc
        
    def matrixelements(self,finitedifference_operator):
        """
        Returns the matrix elements of the current element to its neighbours.

        finitedifference_operator: The discretized operator of the differential equation.
        """
        if self.potential != None:
            matrixelements=[(self,1)]
            #inhomogeneity=self.potential
            inhomogeneityelements=0 #Variable is not used
        elif self.neumannbc!=None:
            elements=self.slope_operator_1st_order(self.neumannbc[1],finitedifference_operator)
            matrixelements=[x for x in elements if x[0].potential==None]
            
            inhomogeneityelements=0 #Variable is not used
        else:               
            elements=self.pure_operator(finitedifference_operator)
            matrixelements=[x for x in elements if x[0].potential==None]
            inhomogeneityelements=[x for x in elements if x[0].potential!=None] #Implicitly assumes that all elements outside the calculation area have the potential 0
            #inhomogeneity=self.charge/self.epsilon
            #for x,v in inhomogeneityelements:
            #    inhomogeneity-=x.potential*v####################################################
         
        
        self.__inhomogeneityelements=inhomogeneityelements
        self.__matrixelements=matrixelements
        
        self.inhomogeneity()
            
        return self.__matrixelements,self.__inhomogeneity        
    
    def inhomogeneity(self): #Implicitly assumes that all elements outside the calculation area have the potential 0, because it just doesn't handle the outside area
        """
        Returns the inhomogeneity of the element.
        """
        if self.potential==None:
            inhomogeneity=self.charge/Constants.epsilon0
            if self.neumannbc==None:
                for x,v in self.__inhomogeneityelements:
                    inhomogeneity-=x.potential*v
            else:
                inhomogeneity=self.neumannbc[0]
        else:
            inhomogeneity=self.potential
        
        self.__inhomogeneity=inhomogeneity
        
        return self.__inhomogeneity
    
    def slope_operator_1st_order(self,direction,finitedifference_operator):
        #there is no epsilon dependency implemented!!
        def my_neighbours(di,dj):
            return self.rect.neighbour(self.i,self.j,di,dj)
 
        denominator=finitedifference_operator.dx*finitedifference_operator.dy #Probably wrong if dx!=dy - check!! WHY BOTH????
        
        if direction=='xb':
            elements=[(my_neighbours(-1,0),-1/denominator),
                      (my_neighbours(0,0),1/denominator)]
        if direction=='xf':
            elements=[(my_neighbours(1,0),1/denominator),
                      (my_neighbours(0,0),-1/denominator)]     
        if direction=='yb':
            elements=[(my_neighbours(0,-1),-1/denominator),
                      (my_neighbours(0,0),1/denominator)]
        if direction=='yf':
            elements=[(my_neighbours(0,1),1/denominator),
                      (my_neighbours(0,0),-1/denominator)]                       
            
        eps=[x[0].epsilon if x[0] is not None else 1 for x in elements] #is not implemented!!!!!
        return [x for x in elements if x[0] != None]
            
    def pure_operator(self,finitedifference_operator):
        def my_neighbours(di,dj):
            return self.rect.neighbour(self.i,self.j,di,dj)
            
        return finitedifference_operator.matrixelements(my_neighbours)
                
    def index(self):
        """
        Return the index of the element in its rectangle.
        """
        return self.rect.pos_to_index(self.i,self.j)
        
    
class FiniteDifferenceOperator:
    """
    Abstract basis class for finite difference operators.
    A class derived from it has to supply variables for dx and dy
    and a function which returns the matrix elements between the
    current basis element and its neighbours (see e.g.
    Laplacian2D2ndOrderWithMaterials)
    """
    dx=0
    dy=0
    matrixelements=0
    
class Laplacian2D2ndOrderWithMaterials(FiniteDifferenceOperator):
    """
    2nd order discreticed Laplace operator in two dimensions for a
    electrostatic problem with dielectric materials.
    """

    def __init__(self,dx,dy):
        """
        Default constructor, dx and dy are the length and width of
        the element.
        """
        self.dx=dx
        self.dy=dy
                    
    def matrixelements(self,my_neighbours):
        """
        Returns the matrix elements between the main element (i,j)
        and its neighbours, which are:

        (i,j) -4
        (i+1,j) 1
        (i-1,j) 1
        (i,j+1) 1

        ...divided by dx*dy and with a factor describing the dielectric property.

        my_neighbours: function which returns the element object of a neighbour of the
                       current element, e.g. my_neighbours(0,0) gibts the current element,
                       my_neighbours(1,0) the one under it etc.

             eps1
        eps2  .  eps3
             eps4

        """
    
        neighbours=[my_neighbours(0,0),my_neighbours(1,0),my_neighbours(-1,0),my_neighbours(0,1),my_neighbours(0,-1)]
        
        eps=[x.epsilon if x is not None else 1 for x in neighbours]
        
        denominator=self.dx*self.dy
        
        return [x for x in [(neighbours[0],(-eps[0]-eps[0]-eps[4]-eps[2])/denominator),
                (neighbours[1],eps[0]/denominator),
                (neighbours[2],eps[2]/denominator),
                (neighbours[3],eps[0]/denominator),
                (neighbours[4],eps[4]/denominator)] #Probably wrong if dx!=dy - check!!
                if x[0] != None
        ]
        
        
class Laplacian2D2ndOrder(FiniteDifferenceOperator):
    """
    2nd order discreticed Laplace operator in two dimensions.
    """
    def __init__(self,dx,dy):
        """
        Default constructor, dx and dy are the length and width of
        the element.
        """
        self.dx=dx
        self.dy=dy
                    
    def matrixelements(self,my_neighbours):    
        """
        Returns the matrix elements between the main element (i,j)
        and its neighbours, which are:

        (i,j) -4
        (i+1,j) 1
        (i-1,j) 1
        (i,j+1) 1

        ...divided by dx*dy.

        my_neighbours: function which returns the element object of a neighbour of the
                       current element, e.g. my_neighbours(0,0) gibts the current element,
                       my_neighbours(1,0) the one under it etc.
        """
        neighbours=[my_neighbours(0,0),my_neighbours(1,0),my_neighbours(-1,0),my_neighbours(0,1),my_neighbours(0,-1)]
        denominator=self.dx*self.dy
        return [x for x in [(neighbours[0],-4./denominator),
                (neighbours[1],1./denominator),
                (neighbours[2],1./denominator),
                (neighbours[3],1./denominator),
                (neighbours[4],1./denominator)] #Probably wrong if dx!=dy - check!!
                if x[0] != None
        ]
    
class Rectangle:
    """
    Rectangle describes a rectangular geometry/grid, containing of mxn elements.
    It goes through all its element objects and creates the matrices and inhomogeneities,
    according to the geometry and the boundary conditions, described in the elements.
    """
    elementlist=0
    finitedifference_operator=0
    m=0
    n=0
    #connected_rectangles=0
    container=0
    
    def __init__(self,m,n,epsilon,finitedifference_operator,fermi_energy_charge_dependence=None):
        """
        m: Number of rows
        n: Number of columns
        epsilon: Relative dielectric constant
        finitedifference_operator: The operator of the differential equation.
        fermi_energy_charge_dependence: How the fermi energy of the material depends on the charge. Default is None.
                                        Mind that this setting assumes that the element is in a homogeneous environment.
        """
        self.m=m
        self.n=n
        self.elementlist=[Element(self,i,j,epsilon=epsilon,fermi_energy_charge_dependence=fermi_energy_charge_dependence) for i in range(m) for j in range(n)]
        self.finitedifference_operator=finitedifference_operator   
    

    def neighbour(self,i,j,di,dj):
        """
        Given coordinates i,j, find out who the neighbour in di,dj direction is.
        """        
        if i+di>=0 and i+di<self.m and j+dj>=0 and j+dj<self.n:
            return self[i+di,j+dj]
        else:
            for rect,offsets in self.container.rectangle_connections[self].items():
                for offset in offsets:
                    if i+di>=offset[0] and i+di<offset[0]+rect.m and j+dj>=offset[1] and j+dj<offset[1]+rect.n:
                        return rect[i+di-offset[0],j+dj-offset[1]]
            return None
        
    def pos_to_index(self,i,j):
        """
        Calculate the index of an element at a given position (i,j).
        The index is a number running from top to bottom, from left to right.
        """
        if i<self.m and j<self.n:
            return self.n*i+j
        else:
            raise KeyError("The Rectangle is smaller than the given coordinates.")
    
    def __getitem__(self,x):
        """
        Returns the element at the given index i,j.

        Example:
        element=my_rectangle[3,4]
        """
        return self.elementlist[self.pos_to_index(x[0],x[1])]

    def creatematrices(self):
        """
        Create matrices. The rectangle may be connected with other
        rectangles (the Container class takes care of that).
        The function creates the matrices for the interaction with itself
        and with every other rectangle it is connected to.
        """
        matrices={}
        inhomogeneity=numpy.zeros(len(self.elementlist))
        for other_rect in self.container.rectangle_connections[self].keys():
            matrices[other_rect]=scipy.sparse.lil_matrix((len(self.elementlist),len(other_rect.elementlist)))
        for element in self.elementlist:
            idx1=element.index()
            matrixelements,inhom=element.matrixelements(self.finitedifference_operator)
            inhomogeneity[idx1]=inhom
            for elem,val in matrixelements:
                #if matrices[elem.rect][idx1,elem.index()]!=0.:
                #    print "already nonzero " + str(idx1)+" "+str(elem.index())
                matrices[elem.rect][idx1,elem.index()]+=val #The += is needed when an element influences itself, 
                                                            #then a non-diagonal element is added to a diagonal element here.
                                                            #OR if an element is influenced by an other element twice, e.g. from left and right.                    
                                                            #This can be the case for periodic boundary conditions.
        return matrices,inhomogeneity
    
    def createinhomogeneity(self):
        """
        Create the inhomogeneity.
        """
        inhomogeneity=numpy.zeros(len(self.elementlist))
        for element in self.elementlist:
            inhom=element.inhomogeneity()
            inhomogeneity[element.index()]=inhom
        return inhomogeneity
    
class PeriodicContainer:
    """
    Contains a single rectangle which is periodically repeated in one direction.
    (by placing copies of itself next to it).
    """
    rectangle_list=None
    rectangle_connections=None
    
    def __init__(self,rectangle,mode='x'):
        """
        rectangle: The rectangle to repeat.
        mode: 'x': The rectangle is repeated in x direction only (default).
              'y': The rectangle is repeated in y direction only.
              'xy':The rectangle is repeated in x and y direction.
        """
        
        self.rectangle_list=[rectangle]        
        self.connect(mode)
        
    def connect(self,mode):
        rectangle=self.rectangle_list[0]
        
        self.rectangle_connections=collections.defaultdict(collections.defaultdict)
        self.rectangle_connections[rectangle][rectangle]=[[0,0]]
        rectangle.container=self
            
        if mode=='x' or mode=='xy':
            self.rectangle_connections[rectangle][rectangle].append([-rectangle.m,0])
            self.rectangle_connections[rectangle][rectangle].append([rectangle.m,0])
        
        if mode=='y' or mode=='xy':
            self.rectangle_connections[rectangle][rectangle].append([0,rectangle.n])
            self.rectangle_connections[rectangle][rectangle].append([0,-rectangle.n])
            
###########################################################################
###########################################################################
###########################################################################
#Kopie aus Container

    def rectangle_elementnumbers_range(self):
        nrrange={}
        ctr=0
        for rec in self.rectangle_list:
            nrrange[rec]=(ctr,ctr+rec.m*rec.n)
            ctr+=rec.m*rec.n        
        return nrrange        
        
    def creatematrix(self):
        """
        Create matrix for the whole container. The solution of the differential
        equation 
        """
        matrixarray={}
        inhomarray={}
        for rec in self.rectangle_list:
            matrixarray[rec],inhomarray[rec]=rec.creatematrices()
        supermatrix=scipy.sparse.bmat([[matrixarray[rec][other_rec] for other_rec in self.rectangle_list] for rec in self.rectangle_list],format='csc')
        inhomogeneity=numpy.concatenate([inhomarray[rec] for rec in self.rectangle_list])
        return supermatrix,inhomogeneity
    
    def createinhomogeneity(self):
        """
        Create inhomogeneity for the whole container.
        You can change the boundary condition values (e.g. different voltage) and create
        the new inhomogeneity.
        """
        return numpy.concatenate([rec.createinhomogeneity() for rec in self.rectangle_list])  
    
    def vector_to_datamatrix(self,vec):
        """
        Creates a data matrix out of a solution vector of this system that can be plotted
        using imshow().

        vec: Vector that is a solution for this system.

        Return:
        datamatrix: Matrix, plottable with imshow().
        extent: Plot range parameter for imshow().

        Example:
        datamatrix,extent = my_container.vector_to_datamatrix(vec)
        imshow(data,extent=extent)
        """
        
        abs_pos={self.rectangle_list[0]:(0,0)}
        
        imin,imax,jmin,jmax=0,0,0,0
        
        for rect in self.rectangle_list:
            for other_rect,offsets in self.rectangle_connections[rect].items():
                offset=offsets[0] #Only "first" position of each rectangle will be considered for plot
                abs_pos[other_rect]=abs_pos[rect][0]+offset[0],abs_pos[rect][1]+offset[1]
        
        for rect,pos in abs_pos.items():
            imin,imax=min(imin,pos[0]),max(imax,pos[0]+rect.m)
            jmin,jmax=min(jmin,pos[1]),max(jmax,pos[1]+rect.n)
            
        extent=jmin,jmax,imax,imin
        
        datamatrix=numpy.ones((imax-imin,jmax-jmin))*numpy.nan
        
        rectangle_elementnumbers_range=self.rectangle_elementnumbers_range()
        for rect,pos in abs_pos.items():
            elements=rectangle_elementnumbers_range[rect]
            datamatrix[pos[0]-imin:pos[0]-imin+rect.m,
                       pos[1]-jmin:pos[1]-jmin+rect.n]=vec[elements[0]:elements[1]].reshape(rect.m,rect.n)
        
        return datamatrix,extent
    
    def simple_plot(self,vec):
        """
        Create a simple plot of the solution.
        """
        fig=pylab.figure()
        ax = fig.gca()
        
        datamatrix,extent=self.vector_to_datamatrix(vec)
        pl=ax.imshow(datamatrix,extent=extent)
        fig.colorbar(pl, shrink=0.9, aspect=3)
        
    def solve_and_plot(self):
        """
        Solve the system and create a simple plot.        
        """
        solve,inhom=self.lu_solver()
        
        x=solve(inhom)
        self.simple_plot(x)
        
    def apply_operator(self,vec,finitedifference_operator,elements=None):
        """
        vec: Solution vector to apply the operator onto.
        finitedifference_operator: The operator.
        elements: Specific elements to apply the operator onto. If None, it is applied
        to all elements.

        If the operator includes points which are not within the calculated area
        (=rectangle + those connected to it), they are implicitly assumed to be zero.

        Return:
        result: Result of the operator on the vector. If elements=None (=all elements),
        this can be plotted with simple_plot().
        """
        
        if elements==None:
            elements=[]
            for rect in self.rectangle_list:
                elements+=rect.elementlist
                
        rectangle_elementnumbers_range=self.rectangle_elementnumbers_range()
                
        def apply(pure_operator):
            r=0
            for elem,val in pure_operator:
                r+=val*vec[elem.index()+rectangle_elementnumbers_range[elem.rect][0]]
            return r
        
        result = numpy.array([apply(elem.pure_operator(finitedifference_operator)) for elem in elements])
        return result
    
    def charge(self,vec,finitedifference_operator,elements=None):
        """
        Calculate the charge with a given operator.
        This is a wrapper for apply_operator() which additionally multiplies with \epsilon_0.
        """
        return self.apply_operator(vec,finitedifference_operator,elements)*Constants.epsilon0
        
    def get_values_at_elements(self,vec,elements):
        """
        Get values of given elements in solution vector.
        
        vec: solution vector
        elements: list of elements to get the solution at
        """
        
        rectangle_elementnumbers_range=self.rectangle_elementnumbers_range()
        
        return [vec[elem.index()+rectangle_elementnumbers_range[elem.rect][0]] 
         for elem in elements]
        
    def lu_solver(self):
        """
        Create the system matrix and solve the system by LU decomposition.

        Return:
        solver: Function that gives the solution x for a given inhomogenity b.
        inhomogeneity: Inhomogeneity of the current configuration.

        Example:
        solver,inhomogeneity=my_container.lu_solver()
        x=solver(inhomogeneity)
        """
        
        matrix,inhomogeneity=self.creatematrix()
        solve=scipy.sparse.linalg.factorized(matrix)
        
        return solve,inhomogeneity
###########################################################################
###########################################################################
###########################################################################

class Container:
    """
    Container contains one or more rectangles and is responsible for gathering
    the submatrices and sub-inhomogeneities created by the Rectangle objects,
    putting them into one matrix/vector and solving the system.
    """
    rectangle_list=0
    rectangle_connections=0
    
    def connect(self,rect,other_rect,align='top',position='right',offset=(0,0),viceversa=True):
        """
        The connect function sets the relationship of one rectangle in the container to an other
        rectangle.

        align: How the other rectangle is aligned relative to the first rectangle.
        Possible values are 'top','bottom','left' and 'right'.
        position: Position of the other rectangle relative to the first rectangle.
        Possible values are 'top','bottom','left' and 'right'.
        offset: offset vector in lattice units, starting from the position given by align and position
        viceversa: equally connect other rectangle automatically. Default is True.

        Examples: align='top',position='right': the tops of the rectangles will be aligned, and the other
                  rectangle is right of the current one.
                  align='right',position='bottom': the right sides of the rectangles will be aligned,
                  and the other rectangle is below the first rectangle.

        align='top' or 'bottom' have to be combined with position='right' or 'left' and vice versa!
        """
        totaloffset=[0,0]
        if align=='top':
            totaloffset[0]=0
        if align=='bottom':
            totaloffset[0]=rect.m-other_rect.m
        if align=='left':
            totaloffset[1]=0
        if align=='right':
            totaloffset[1]=rect.n-other_rect.n
        if position=='top':
            totaloffset[0]=-other_rect.m
        if position=='bottom':
            totaloffset[0]=rect.m
        if position=='left':
            totaloffset[1]=-other_rect.n
        if position=='right':
            totaloffset[1]=rect.n
            
        totaloffset[0]+=offset[0]
        totaloffset[1]+=offset[1]
            
        self.rectangle_connections[rect][other_rect]=[totaloffset]
        
        if viceversa:
            if position=='top':
                other_position='bottom'
            if position=='bottom':
                other_position='top'
            if position=='left':
                other_position='right'
            if position=='right':
                other_position='left'                
            
            other_offset=-offset[0],-offset[1]
            self.connect(other_rect,rect,align=align,position=other_position,offset=other_offset,viceversa=False)        

    def __init__(self,rectangle_list):
        """ 
        rectangle_list: List of rectangles participating in the calculation.

        Invoke connect() afterwards to set the connection between the rectangles.
        """
        self.rectangle_list=list(rectangle_list)
        
        self.rectangle_connections=collections.defaultdict(collections.defaultdict)
        for rect in self.rectangle_list:
            self.rectangle_connections[rect][rect]=[[0,0]]
            rect.container=self
    
    def add_rectangle(self,rect):
        """
        Add an additional rectangle.
        """
        self.rectangle_list.append(rect)
        self.rectangle_connections[rect][rect]=[[0,0]]
        rect.container=self
        
    def rectangle_elementnumbers_range(self):
        nrrange={}
        ctr=0
        for rec in self.rectangle_list:
            nrrange[rec]=(ctr,ctr+rec.m*rec.n)
            ctr+=rec.m*rec.n        
        return nrrange        
        
    def creatematrix(self):
        """
        Create matrix for the whole container.
        """
        matrixarray={}
        inhomarray={}
        for rec in self.rectangle_list:
            matrixarray[rec],inhomarray[rec]=rec.creatematrices()
        supermatrix=scipy.sparse.bmat([[matrixarray[rec][other_rec] for other_rec in self.rectangle_list] for rec in self.rectangle_list],format='csc')
        inhomogeneity=numpy.concatenate([inhomarray[rec] for rec in self.rectangle_list])
        return supermatrix,inhomogeneity
    
    def createinhomogeneity(self):
        """
        Create inhomogeneity for the whole container.
        You can change the boundary condition values (e.g. different voltage) and create
        the new inhomogeneity.
        """
        return numpy.concatenate([rec.createinhomogeneity() for rec in self.rectangle_list])  
    
    def vector_to_datamatrix(self,vec):
        """
        Creates a data matrix out of a solution vector of this system that can be plotted
        using imshow() or used for export.

        vec: Vector that is a solution for this system.

        Return:
        datamatrix: Matrix, plottable with imshow().
        extent: Plot range parameter for imshow().

        Example:
        datamatrix,extent = my_container.vector_to_datamatrix(vec)
        imshow(data,extent=extent)
        """
        
        abs_pos={self.rectangle_list[0]:(0,0)}
        
        imin,imax,jmin,jmax=0,0,0,0
        
        for rect in self.rectangle_list:
            for other_rect,offsets in self.rectangle_connections[rect].items():
                offset=offsets[0] #Only "first" position of each rectangle will be considered for plot
                abs_pos[other_rect]=abs_pos[rect][0]+offset[0],abs_pos[rect][1]+offset[1]
        
        for rect,pos in abs_pos.items():
            imin,imax=min(imin,pos[0]),max(imax,pos[0]+rect.m)
            jmin,jmax=min(jmin,pos[1]),max(jmax,pos[1]+rect.n)
            
        extent=jmin,jmax,imax,imin
        
        datamatrix=numpy.ones((imax-imin,jmax-jmin))*numpy.nan
        
        rectangle_elementnumbers_range=self.rectangle_elementnumbers_range()
        for rect,pos in abs_pos.items():
            elements=rectangle_elementnumbers_range[rect]
            datamatrix[pos[0]-imin:pos[0]-imin+rect.m,
                       pos[1]-jmin:pos[1]-jmin+rect.n]=vec[elements[0]:elements[1]].reshape(rect.m,rect.n)
        
        return datamatrix,extent
    
    def simple_plot(self,vec):
        """
        Create a simple plot of the solution.
        """
        fig=pylab.figure()
        ax = fig.gca()
        
        datamatrix,extent=self.vector_to_datamatrix(vec)
        pl=ax.imshow(datamatrix,extent=extent)
        fig.colorbar(pl, shrink=0.9, aspect=3)
        
    def solve_and_plot(self):
        """
        Solve the system and create a simple plot.
        """        
        solve,inhom=self.lu_solver()
        
        x=solve(inhom)
        self.simple_plot(x)
        
    def apply_operator(self,vec,finitedifference_operator,elements=None):
        """
        vec: Solution vector to apply the operator onto.
        finitedifference_operator: The operator.
        elements: Specific elements to apply the operator onto. If None, it is applied
        to all elements.

        If the operator includes points which are not within the calculated area
        (=rectangle + those connected to it), they are implicitly assumed to be zero.

        Return:
        result: Result of the operator on the vector. If elements=None (=all elements),
        this can be plotted with simple_plot().
        """
        
        if elements==None:
            elements=[]
            for rect in self.rectangle_list:
                elements+=rect.elementlist
                
        rectangle_elementnumbers_range=self.rectangle_elementnumbers_range()
                
        def apply(pure_operator):
            r=0
            for elem,val in pure_operator:
                r+=val*vec[elem.index()+rectangle_elementnumbers_range[elem.rect][0]]
            return r
        
        result = numpy.array([apply(elem.pure_operator(finitedifference_operator)) for elem in elements])
        return result
    
    def charge(self,vec,finitedifference_operator,elements=None):
        """
        Calculate the charge with a given operator.
        This is a wrapper for apply_operator() which additionally multiplies with \epsilon_0.
        """
        return self.apply_operator(vec,finitedifference_operator,elements)*Constants.epsilon0
        
        
    def lu_solver(self):
        """
        Create the system matrix and solve the system by LU decomposition.

        Return:
        solver: Function that gives the solution x for a given inhomogenity b.
        inhomogeneity: Inhomogeneity of the current configuration.

        Example:
        solver,inhomogeneity=my_container.lu_solver()
        x=solver(inhomogeneity)
        """
        
        matrix,inhomogeneity=self.creatematrix()
        solve=scipy.sparse.linalg.factorized(matrix)
        
        return solve,inhomogeneity
