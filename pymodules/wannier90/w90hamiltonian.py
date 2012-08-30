import general
import numpy
import math
import cmath
import numpy
from vasp import poscar
from scipy import linalg
from matplotlib import pyplot

class Hamiltonian:
    
    __unitcellmatrixblocks=[]
    __unitcellnumbers=[]
    #TODO: make a map/dictionary out of those two
    __poscardata=0
    __nrbands=0
    __blochmatrix=0
    
    def __init__(self,wannier90filename,poscarfilename):
        self.__poscardata = poscar.PoscarData(poscarfilename)
        self.__nrbands,wanndata = self.__read_wannier90_hr_file(wannier90filename)
        self.__unitcellmatrixblocks, self.__unitcellnumbers = self.__process_wannier90_hr_data(wanndata)
        
    def __process_wannier90_hr_data(self, wanndata):
        prevcell = []
        unitcells = []
        for line in wanndata:
            currentcell = line[0:2]
            if currentcell != prevcell:
                unitcells.append([])
            unitcells[-1].append(line)
            prevcell = currentcell
        
        unitcellnumbers = [[int(x) for x in unitcell[0][0:3]] for unitcell in unitcells]
        unitcellmatrixblocks = []
        for unitcell in unitcells:
            elementlist = numpy.array([complex(float(line[5]), float(line[6])) for line in unitcell])
            unitcellmatrixblocks.append(elementlist.reshape((self.__nrbands, self.__nrbands)))
        
        return unitcellmatrixblocks, unitcellnumbers

    def __read_wannier90_hr_file(self,filename):
        data=general.read_file_as_table(filename)
        
        nrbands=int(data[1][0])
        linestart=int(math.ceil(float(data[2][0])/15))+3
        wanndata=data[linestart:]
        
        return nrbands,wanndata
        #scipy.linalg.blas.fblas.zaxpy
        
    def __bloch_phases(self,k):
        """
        Calculates the bloch factor e^ikr for each unit cell in
        self.__unitcellnumbers
        """
        latticevecs_transposed=numpy.transpose(self.__poscardata.latticevecs())
        return numpy.array([cmath.exp(complex(0,1)*numpy.dot(k, \
        numpy.dot(latticevecs_transposed,cellnumber))) for cellnumber in self.__unitcellnumbers])
        
    def __unitcellcoordinates_to_nrs(self,usedhoppingcells):
        """
        Given a list of unit cell coordinates, the function
        converts them to integer indices i for __unitcellmatrixblocks[i] and
        __unitcellnumbers[i].
        """
        
        indices=[]
        for cell in usedhoppingcells:
            index=self.__unitcellnumbers.index(cell)
            indices.append(index)
            
        return indices   
    
    def bloch_eigenvalues(self,k,basis='c',usedhoppingcells='all'):
        """
        Calculates the eigenvalues of the eigenvalue problem with
        Bloch boundary conditions for a given vector k.
        
        usedhoppingcells: If you don't want to use all hopping parameters,
        you can set them here (get the list of available cells with unitcellnumbers() and
        strip the list from unwanted cells).
        basis: 'c' or 'd'. Determines if the kpoints are given in cartesian
        reciprocal coordinates or direct reciprocal coordinates.
        """
        
        if usedhoppingcells == 'all':
            usedunitcellnrs=range(len(self.__unitcellnumbers))
        else:
            usedunitcellnrs=self.__unitcellcoordinates_to_nrs(usedhoppingcells)
        
        if basis=='d':
            k=self.__poscardata.direct_to_cartesian_reciprocal(k)

        bloch_phases=self.__bloch_phases(k)
        blochmatrix = numpy.zeros((self.__nrbands, self.__nrbands), dtype=complex)
        
        for i in usedunitcellnrs:
            blochmatrix += bloch_phases[i] * self.__unitcellmatrixblocks[i]

        evals,evecs=linalg.eig(blochmatrix)
        return numpy.sort(evals.real)   
    
    def bandstructure_data(self,kpoints,basis='c',usedhoppingcells='all'):
        """
        Calculates the bandstructure for a given kpoint list.
        For direct plotting, use plot_bandstructure(kpoints,filename).
                
        A list of eigenvalues for each kpoint is returned. To sort 
        by band, use data.transpose().
        
        usedhoppingcells: If you don't want to use all hopping parameters,
        you can set them here (get the list of available cells with unitcellnumbers() and
        strip the list from unwanted cells).        
        basis: 'c' or 'd'. Determines if the kpoints are given in cartesian
        reciprocal coordinates or direct reciprocal coordinates.
        """
            
        data=numpy.array([self.bloch_eigenvalues(kpoint,basis,usedhoppingcells) for kpoint in kpoints])
        return data
    
    def point_path(self,corner_points,nrpointspersegment):
        """
        Generates a path connecting the corner_points with nrpointspersegment points per segment
        (excluding the next point), resulting in sum(nrpointspersegment)+1 points.
        The points in corner_points can have any dimension.
        nrpointspersegment is a list with one element less than corner_points.
        If nrpointspersegment is an integer, it is assumed to apply to each segment.
        
        Example: 
        my_hamiltonian.point_path([[0,0],[1,1],[2,2]],[2,2])
        gives        
        [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [1.5, 1.5], [2, 2]]
        (note: those are 5 points, which is sum([2,2])+1)
        
        or equivalently:
        my_hamiltonian.point_path([[0,0],[1,1],[2,2]],2)
        """
        """TODO: maybe put this function somewhere else?"""

        if type(nrpointspersegment) != int and len(corner_points) != len(nrpointspersegment)+1:
            raise ValueError('corner_points has to be one element larger than nrpointspersegment, unless nrpointspersegment is integer.')
        if type(nrpointspersegment)==int:
            nrpointspersegment=(len(corner_points)-1)*[nrpointspersegment]
        if type(corner_points[0])!=list:
            corner_points=[[x] for x in corner_points]

        points=[]
        for i in range(len(nrpointspersegment)):
            newpoints=self.__path_between_two_vectors(corner_points[i], corner_points[i+1], nrpointspersegment[i])
            points.extend(newpoints)
            
        points.append(corner_points[-1])
            
        return points
    
    def __path_between_two_vectors(self,v1,v2,nrpoints):
        """
        Generates a path between v1 and v2 (lists of any dimension) with nrpoints elements.
        The last point, v2, is not in the path.
        """
        dimension=len(v1)
        return numpy.transpose([numpy.linspace(v1[j], \
                v2[j],nrpoints,endpoint=False) for j in range(dimension)]).tolist()
        
    def plot_bandstructure(self,kpoints,filename,basis='c',usedhoppingcells='all'):
        """
        Calculate the bandstructure at the points kpoints (given in 
        cartesian reciprocal coordinates - use direct_to_cartesian_reciprocal(k)
        if you want to use direct coordinates) and save the plot
        to filename. The ending of filename determines the file format.
        
        usedhoppingcells: If you don't want to use all hopping parameters,
        you can set them here (get the list of available cells with unitcellnumbers() and
        strip the list from unwanted cells).  
        basis: 'c' or 'd'. Determines if the kpoints are given in cartesian
        reciprocal coordinates or direct reciprocal coordinates.
        """

        data=self.bandstructure_data(kpoints,basis,usedhoppingcells)
        bplot=BandstructurePlot()
        bplot.plot(kpoints, data)
        bplot.save(filename)
                
    def unitcellnumbers(self):
        """
        Returns the numbers of the unit cells supplied in the wannier90_hr.dat
        file.
        """
        return list(self.__unitcellnumbers) #makes a copy instead of a reference
    
class BandstructurePlot:
    """
    Combine several bandstructure plots.
    
    Call plot(kpoints,data) for every bandstructure plot.
    Then, call save(filename) to save to a file.
    """
    
    __stylelist=['b-','g-','r-','c-','m-','y-','k-']
    __plotcounter=0

    def __init__(self):
        pyplot.clf()
        
    def setstyles(self,stylelist):
        """
        Set the styles for each bandstructure plot. 
        
        stylelist: A list with format strings (see 
        http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.plot)
        for each plot. If the list is shorter than the number of plots,
        it is repeated.
        """
        
        self.__stylelist = stylelist
        
        
    def __kpoints_to_pathlength(self,points):       
        """
        points: list of points
        
        Calculates the distance from the first point to each point in the list along the path.
        """
        points=numpy.array(points)
        distances=[0]
        distance=0
        for prev,this in zip(points[:-1],points[1:]):
            distance=numpy.linalg.norm(this-prev)+distance
            distances.append(distance)
        return distances
    
    def plot(self,kpoints,data,style='auto'):
        """
        Add a bandstructure plot to the figure.
        
        kpoints: list of kpoints
        data: list of eigenvalues for each kpoint.
        style: set format string of this plot. You can also set the 
        styles of all plots using setstyles().
        Default is 'auto', which means that the style list will be used.
        """
        pathlength=self.__kpoints_to_pathlength(kpoints)
        if style == 'auto':
            stylestring=self.__stylelist[self.__plotcounter % len(self.__stylelist)]
        else:
            stylestring=style
            
        self.__plotcounter+=1
        for band in data.transpose():
            pyplot.plot(pathlength,band,stylestring)
    
    def save(self,filename):
        """
        Save the figure to a file. The format is determined
        by the filename.
        """
        pyplot.savefig(filename,dpi=(150))
        
    def reset(self):
        """
        Clear the current figure.
        """
        pyplot.clf()
        
    def show(self):
        pyplot.show()
        
