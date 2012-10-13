import general
import numpy
import math
import cmath
from vasp import poscar
from scipy import linalg
from matplotlib import pyplot
from matplotlib.path import Path
import matplotlib.patches as patches
import itertools

class Hamiltonian:
    
    
    """
    TODO: durchschleifen der argumente bei bloch_eigenvalues etc. ist bloed. vl. argumente bei allen anderen
    mit *args und auf dokumentation von bloch_eigenvalues verweisen?
    TODO: unitcellcoordinates and numbers is used ambiguously
    TODO: performance: use in-place operations for numpy -=, +=, *= and consider numpy.fromfunction
    TODO: to ensure a variable is a numpy array: a = array(a, copy=False)
    TODO: read in fermi energy and make it possible to set it to 0
    TODO: REFACTOR REFACTOR REFACTOR

    """
    
    __unitcellmatrixblocks=[]
    __unitcellnumbers=[]
    __orbitalspreads=[]
    __orbitalpositions=[]
    #TODO: make a map/dictionary out of those two
    __latticevecs=0
    __nrbands=0
    
    def __init__(self):
        """
        There are two ways to initialize the Wannier90 Hamiltonian:
        1) Hamiltonian.from_file(wannier90filename,poscarfilename,wannier90woutfilename)
        2) Hamiltonian.from_raw_data(unitcellmatrixblocks,unitcellnumbers,latticevecs)
        
        See the documentation of those methods.
        """   
        
        pass 
        
        #TODO: wannier90filename should be the id of the wannier90 calculation, and specific
        #filenames derived from that ('bla' -> bla.win, bla.wout, bla_hr.dat etc.). Then,
        #kick out all filename method arguments
        
    def latticevectors(self):
        """
        Return the system's lattice vectors.
        """
        
        return numpy.array(self.__latticevecs.latticevecs()) #copy
    
    def reciprocal_latticevectors(self):
        """
        Return the system's reciprocal lattice vectors.
        """
        
        return numpy.array(self.__latticevecs.reciprocal_latticevecs()) #copy
    
    def orbitalspreads(self):
        """
        Return the wannier90 orbital spreads.
        """
        
        return list(self.__orbitalspreads)
    
    def orbitalpositions(self):
        """
        Return the wannier90 orbital positions.
        """
        
        return list(self.__orbitalpositions)
    
        
    @classmethod
    def from_file(cls,wannier90filename,poscarfilename,wannier90woutfilename):
        """
        A constructor to create an object based on data from files.
        wannier90filename: Path to the wannier90_hr.dat file
        poscarfilename: Path to the VASP POSCAR file
        wannier90woutfilename: Path to the wannier90.wout file
        """        
        self = cls()
        
        poscardata = poscar.PoscarData(poscarfilename)
        self.__latticevecs=poscardata.lattice_vectors
        self.__nrbands,wanndata = self.__read_wannier90_hr_file(wannier90filename)
        self.__unitcellmatrixblocks, self.__unitcellnumbers = self.__process_wannier90_hr_data(wanndata)
        self.__orbitalspreads,self.__orbitalpositions=self.__orbital_spreads_and_positions(wannier90woutfilename)
        
        return self
    
    @classmethod
    def from_raw_data(cls,unitcellmatrixblocks,unitcellnumbers,latticevecs,orbitalspreads,orbitalpositions):
        """
        A constructor used to create a custom Hamiltonian.
        unitcellmatrixblocks: Hopping elements, arranged by unit cells.
        unitcellnumbers: coordinates of the unit cells "hopped" to.
        latticevecs: lattice vectors.
        orbitalspreads: spreads of the orbitals
        orbitalpositions: positions of the orbitals
        """              
        self = cls()
        
        self.__unitcellnumbers = unitcellnumbers
        self.__unitcellmatrixblocks = unitcellmatrixblocks
        self.__latticevecs = poscar.LatticeVectors(latticevecs)
        self.__nrbands = len(unitcellmatrixblocks[0])
        self.__orbitalspreads=orbitalspreads
        self.__orbitalpositions=orbitalpositions
        
        return self
        
    def __process_wannier90_hr_data(self, wanndata):
        prevcell = []
        unitcells = []
        for line in wanndata:
            currentcell = line[0:3]
            if currentcell != prevcell:
                unitcells.append([])
            unitcells[-1].append(line)
            prevcell = currentcell
        
        unitcellnumbers = [[int(x) for x in unitcell[0][0:3]] for unitcell in unitcells]
        unitcellmatrixblocks = []
        for unitcell in unitcells:
            elementlist = numpy.array([complex(float(line[5]), float(line[6])) for line in unitcell])
            unitcellmatrixblocks.append(numpy.transpose(elementlist.reshape((self.__nrbands, self.__nrbands)))) #Transpose because first index in wannier90_hr.dat file runs faster than second
        
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
        #TODO: if k is direct: lattice vectors are probably not necessary. How could that work?
        latticevecs_transposed=numpy.transpose(self.__latticevecs.latticevecs())
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
    
    def __sorting_order(self,data,key=None):
        """
        Sorts a list by the given column sortcolumn and returns the order. 
        If key==None (default), all columns are sorted, with the last column running fastest.
        """
        if key==None:
            return [i[0] for i in sorted(enumerate(data),key=lambda x: x[1])]
        else:
            return [i[0] for i in sorted(enumerate(data),key=lambda x: key(x[1]))]
    
    
    def __apply_order(self,data,order):
        """Applies an order to a list."""
        return [data[i] for i in order]
    
    def write_matrix_elements(self,outputfile,usedhoppingcells='all',usedorbitals='all'):
        """
        Write the wannier90 matrix elements to a file (*.wetb) readable by Florian's code.
        Information contained in the file:
        - lattice vectors
        - orbital positions
        - orbital spreads
        - matrix elements of the chosen unit cells and orbitals
        
        Description of the file format:
        The file contains three sections, divided by an arbitrary number of blank lines (one at least,
        obviously). Lines starting with # are comments and are ignored. Anything in a line after the
        data is also ignored (i.e. you can write anything in the same line after the data, with or
        without #).
        First block: lattice vectors in rows
        Second block: orbital spread (first column) and position (other columns) of every orbital
        Third block: Matrix elements.
            Column 1-3: Unit cell number
            Column 4: Orbital number in main unit cell
            Column 5: Orbital number in other unit cell (the one the electron "hops" to)
            Column 6&7: Real & imaginary part of the matrix element
        
        outputfile: Name of the output file (*.wetb - Wannier90-Environmental-dependent-Tight-Binding)
        usedhoppingcells: If you don't want to use all hopping parameters,
        you can set them here (get the list of available cells with unitcellnumbers() and
        strip the list from unwanted cells).
        usedorbitals: a list of used orbitals to use. Default is 'all'. Note: this only makes
        sense if the selected orbitals don't interact with other orbitals.
        """
        
        output=open(outputfile,'w')
        
        if usedhoppingcells == 'all':
            usedunitcellnrs=range(len(self.__unitcellnumbers))
        else:
            usedunitcellnrs=self.__unitcellcoordinates_to_nrs(usedhoppingcells)
            
        if usedorbitals=='all':
            orbitalnrs=range(self.__nrbands)
        else:
            orbitalnrs=usedorbitals
            
        order_usedunitcellnumbers=self.__apply_order(usedunitcellnrs, self.__sorting_order([self.__unitcellnumbers[i] for i in usedunitcellnrs]))
            
        output.write('#WETB File\n\n')
        output.write('#Lattice vectors:\n')
        latticevecs=self.__latticevecs.latticevecs()
        for vec in latticevecs:
            output.write('{:12.6f} {:12.6f} {:12.6f}\n'.format(*vec))
        output.write('\n\n')
        output.write('#Spreads and positions of the orbitals:\n')
        spreads=self.__orbitalspreads
        positions=self.__orbitalpositions
        for spread,position in zip(spreads,positions):
            output.write('{:12.6f} '.format(spread))
            output.write('{:12.6f} {:12.6f} {:12.6f}\n'.format(*position))
        output.write('\n\n')
        
        output.write('#Unit cell number, main orbital, other orbital, hopping element:\n')
        for i in order_usedunitcellnumbers:
            for j, val in numpy.ndenumerate(self.__unitcellmatrixblocks[i][numpy.ix_(orbitalnrs,orbitalnrs)]):
                #j: ndenumerate assigns the hop-from and hop-to number to each hopping matrix element
                
                #http://docs.python.org/library/string.html#format-specification-mini-language
                cellnrs=self.__unitcellnumbers[i]
                output.write('{:5d} {:5d} {:5d} '.format(*cellnrs))
                output.write('{:5d} {:5d} '.format(*j))
                output.write('{:12.6f} {:12.6f}\n'.format(val.real,val.imag))
                            
                
        
        output.close()
    
    
    def maincell_eigenvalues(self,usedorbitals='all'):
        """
        Calculates the eigenvalues of the main cell (no hopping to adjacent unit cells).
        
        usedorbitals: a list of used orbitals to use. Default is 'all'. Note: this only makes
        sense if the selected orbitals don't interact with other orbitals.
        """
        
        if usedorbitals=='all':
            orbitalnrs=range(self.__nrbands)
        else:
            orbitalnrs=usedorbitals
        if usedorbitals=='all':
            blochmatrix = self.__unitcellmatrixblocks[self.__unitcellcoordinates_to_nrs([[0,0,0]])[0]]
        else:
            blochmatrix = self.__unitcellmatrixblocks[self.__unitcellcoordinates_to_nrs([[0,0,0]])[0]][numpy.ix_(orbitalnrs,orbitalnrs)]
        
        evals,evecs=linalg.eig(blochmatrix)
        return numpy.sort(evals.real)            
    
    def bloch_eigenvalues(self,k,basis='c',usedhoppingcells='all',usedorbitals='all'):
        """
        Calculates the eigenvalues of the eigenvalue problem with
        Bloch boundary conditions for a given vector k.
        
        usedhoppingcells: If you don't want to use all hopping parameters,
        you can set them here (get the list of available cells with unitcellnumbers() and
        strip the list from unwanted cells).
        basis: 'c' or 'd'. Determines if the kpoints are given in cartesian
        reciprocal coordinates or direct reciprocal coordinates.
        usedorbitals: a list of used orbitals to use. Default is 'all'. Note: this only makes
        sense if the selected orbitals don't interact with other orbitals.
        """
        
        """
        TODO: make it return evecs too (don't forget sorting)
        """
        
        if usedhoppingcells == 'all':
            usedunitcellnrs=range(len(self.__unitcellnumbers))
        else:
            usedunitcellnrs=self.__unitcellcoordinates_to_nrs(usedhoppingcells)
        
        if basis=='d':
            k=self.__latticevecs.direct_to_cartesian_reciprocal(k)
            
        if usedorbitals=='all':
            orbitalnrs=range(self.__nrbands)
        else:
            orbitalnrs=usedorbitals

        bloch_phases=self.__bloch_phases(k)
        blochmatrix = numpy.zeros((len(orbitalnrs), len(orbitalnrs)), dtype=complex)
        
        if usedorbitals=='all':
            for i in usedunitcellnrs:
                blochmatrix += bloch_phases[i] * self.__unitcellmatrixblocks[i]
        else:
            for i in usedunitcellnrs:
                # the same thing, but it saves memory not to use [numpy.ix_(orbitalnrs,orbitalnrs)] if it's not
                # necessary
                #http://stackoverflow.com/questions/4257394/slicing-of-a-numpy-2d-array-or-how-do-i-extract-an-mxm-submatrix-from-an-nxn-ar/4258079#4258079
                blochmatrix += bloch_phases[i] * self.__unitcellmatrixblocks[i][numpy.ix_(orbitalnrs,orbitalnrs)]

        evals,evecs=linalg.eig(blochmatrix)
        return numpy.sort(evals.real)
    
    def bandstructure_data(self,kpoints,basis='c',usedhoppingcells='all',usedorbitals='all',parallelmethod=None,parallelthreads=4):
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
        usedorbitals: a list of used orbitals to use. Default is 'all'. Note: this only makes
        sense if the selected orbitals don't interact with other orbitals.
        parallel: Option to parallelize the band structure over k-points.
            None: No parallelization (default)
            'multiprocessor.pool': Parallelization using the multiprocessor.Pool class
            DOES NOT WORK YET!!!
        parallelthreads: Set to the number of threads if parallel != None. Default is 4.
        """
                
        if parallelmethod==None:
            data=numpy.array([self.bloch_eigenvalues(kpoint,basis,usedhoppingcells,usedorbitals) for kpoint in kpoints])
        if parallelmethod=='multiprocessor.pool':
            data=[]

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
        
    def plot_bandstructure(self,kpoints,filename,basis='c',usedhoppingcells='all',usedorbitals='all'):
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
        usedorbitals: a list of used orbitals to use. Default is 'all'. Note: this only makes
        sense if the selected orbitals don't interact with other orbitals.        
        """

        data=self.bandstructure_data(kpoints,basis,usedhoppingcells,usedorbitals)
        bplot=BandstructurePlot()
        bplot.plot(kpoints, data)
        bplot.save(filename)
        
    def drawunitcells(self,unitcellnumbers='all'):
        """
        Create a plot of a list of unit cells.
        
        unitcellnumbers: Numbers of unit cells to plot.
        Default value is 'all', then unitcellnumbers() is used.
        """
        
        if unitcellnumbers == 'all':
            unitcellnumbers=self.__unitcellnumbers
        
        cellstructure=numpy.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,0]])
        lv=self.__latticevecs.latticevecs()
        unitcellform=numpy.dot(cellstructure,lv)[:,:2]
        cellcoords=self.unitcellcoordinates(unitcellnumbers)[:,:2]
        maincell=unitcellnumbers.index([0,0,0])
        verticeslist=numpy.array([[formpoint+cellcoordinate for formpoint in unitcellform] for cellcoordinate in cellcoords])
        maincellvertices=verticeslist[maincell]
        verticeslist=numpy.delete(verticeslist,maincell,axis=0)   
        
        #http://matplotlib.sourceforge.net/users/path_tutorial.html
        
        codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY,
         ]

        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        
        for verts in verticeslist:
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='white', lw=2)
            ax.add_patch(patch)
        

        path = Path(maincellvertices, codes)
        patch = patches.PathPatch(path, facecolor='orange', lw=2)
        ax.add_patch(patch)
        
        ax.set_xlim(-40,40)
        ax.set_ylim(-40,40)
        pyplot.show()  
        
           
        
    def unitcellcoordinates(self,unitcellnumbers='all'):
        """
        Cartesian coordinates of the given unit cells.
        
        unitcellnumbers: a list of the unit cell numbers. 
        Default value is 'all', then unitcellnumbers() is used.
        """
        
        """
        How the formula works:
        
        The unitcellnumbers are the coordinates of the unit
        cells in the basis spanned by the lattice vectors. A transformation
        to cartesian coordinates is just a basis transformation. The columns of
        the transformation matrix are the lattice vectors in cartesian
        coordinates --> we just have to transpose the list of lattice vectors.
        Instead of applying the transformation matrix to each vector, we apply
        it to all of them at the same time by writing the vectors in the columns
        of a matrix (=transposing the list of unit cell numbers).
        Since the transformed vectors are in the columns of the result matrix,
        we need to transpose that one again.
        
        The formula is now (' = transpose):
        (latticevecs' unitcellnumbers')'
        
        But this is:
        (A'B')'=((BA)')'=BA
        
        -->
        unitcellnumbers latticevecs
        
        So that's the formula!
        
         
        """
        latticevecs = self.__latticevecs.latticevecs()
        
        if unitcellnumbers == 'all':
            unitcellnumbers=numpy.array(self.__unitcellnumbers)
        
        return numpy.dot(unitcellnumbers,latticevecs)
    
    def unitcells_within_zone(self,zone,basis='c',norm_order=2):
        """
        Returns a list of unit cells within a certain area. The function
        is comparing the same point in each cell (e.g. always the bottom left end).
        
        zone: can be a number or a tuple:
            number: radius to include cells within.
            tuple: area to include cells within, in the sense of distance from the origin along a direction.
            
        basis: determines if zone is given in cartesian ('c') or direct ('d') coordinates.
        IMPORTANT: If direct coordinates are used, use integers for zone, not float!
        
        norm_order: if zone is a number (=radius), norm_order is the norm to use (mathematical definition, see 
        http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html). Default is 2 (=Euclidean norm)
        Short version: 2 gives you a "circle", numpy.inf a "square".
        
        Examples:
        Cells within 30 Angstrom:
        unitcells_within_zone(30)
        Cells within a 6x8x1 Angstrom cuboid:
        unitcells_within_zone((3.0,4.0,0.5))
        Cells within a 4x4x4 block in direct coordinates:
        unitcells_within_zone((2,2,2),'d')
        """
        
                
        if type(zone) is tuple:
            if basis == 'd':
                zone=abs(numpy.array(zone))
            if basis == 'c':
                reclattice_transposed_inverted=linalg.inv(self.__latticevecs.reciprocal_latticevecs().transpose()) #matrix to transform from cartesian to direct coordinates
                zone=abs(numpy.dot(reclattice_transposed_inverted,numpy.array(zone)))
            unitcellnrs=[unitcellnr for unitcellnr in self.__unitcellnumbers if numpy.floor(numpy.amin(zone-abs(numpy.array(unitcellnr))))>=0]
        else:
            if basis == 'c':
                unitcellnrs=[unitcellnr for unitcellcoords,unitcellnr 
                             in zip(self.unitcellcoordinates(),self.__unitcellnumbers) 
                             if numpy.linalg.norm(unitcellcoords)<=zone]
            if basis == 'd':    
                unitcellnrs=[unitcellnr for unitcellnr in self.__unitcellnumbers if numpy.linalg.norm(unitcellnr,norm_order)<=zone]
    
        return unitcellnrs
    
    def drop_dimension_from_cell_list(self,dimension,unitcellnumbers='all'):
        """
        Takes a list of unit cell coordinates and drops the x,y or/and z dimension
        from the list - this way you can create a 2D material from a 3D material
        or a 1D material from a 2D material. The function deletes all unit cell numbers
        that have a nonzero entry in that dimension.
        
        dimension: the dimension to drop: 0, 1 or 2. Can also be a list, e.g. (0,1) drops
        the first and second dimension. 
        unitcellnumbers: List of unit cell numbers. Default value is 'all'.
        """
        
        if unitcellnumbers=='all':
            unitcellnumbers=self.unitcellnumbers()
            
        stack=list(unitcellnumbers)
        
        if type(dimension) is int:
            dimension=[dimension]
            
        for i in dimension:
            stack = [x for x in stack if x[i]==0]
        
        return stack
        
        


    def hermitian_hoppinglist(self,unitcellnumbers):
        """
        The function removes unit cells from a list of unit cells whose "parity
        partners" are missing to ensure a Hermitian Bloch matrix.
        
        Return: kept, removed
        
        kept: Kept unit cell numbers
        removed: removed unit cell numbers (just for control purposes)
                
        If hopping to a specific unit cell is not used, one has to make sure
        that the parity inversed unit cell (=the cell with the "negative"
        coordinates") is also dropped.
        That's because the matrix elements of the bloch matrix look like this:
        
        ... + \gamma_i e^ikR + \gamma_i e^-ikR + ...
        
        The sum of the two terms is cos(ikR) and real.
        
        --> The function drops the terms which miss their partner and thus won't
        become real.
        
        Note: It makes sense to remove not only the "parity partner", but all unit
        cells which are identical due to symmetry.
        """
        
        stack = list(unitcellnumbers)
        kept = []
        removed = []
        while len(stack)>0:
            element=stack.pop()
            if element == [-i for i in element]: #true for origin
                kept.append(element)
            else:
                try:
                    index_partner=stack.index([-i for i in element])
                    partner=stack.pop(index_partner)
                    kept.append(element)
                    kept.append(partner)
                except ValueError: #raised if -element does not exist
                    removed.append(element)
        return kept,removed
       
            
    def unitcellnumbers(self):
        """
        Returns the numbers of the unit cells supplied in the wannier90_hr.dat
        file.
        """
        return list(self.__unitcellnumbers) #makes a copy instead of a reference
    
    def nrorbitals(self):
        """
        Returns the number of orbitals/bands.
        """
        return self.__nrbands
    
    def standard_paths(self,name,nrpointspersegment=1):
        """
        Gives the standard path for a Bravais lattice in
        direct reciprocal coordinates.
        
        name: Name of the lattice
        nrpointspersegment: optional; if > 1, a list of intermediate points connecting
        the main points is also returned and can be used for a 
        bandstructure path (nrpointspersegment points per segment).
        Default value: 1
        
        Return:
        points,names(,path)
        
        points: points in the path
        names: names of the points
        (path: path with intermediate points. Only returned if nrpointspersegment is > 1)
        """
        if name=='hexagonal':
            path = [
                    ('G',[0,0,0]),
                    ('K',[1./3,-1./3,0]),
                    ('M',[0.5,0,0]),
                    ('G',[0,0,0])
                    ]
        elif name=='fcc':
            path = [
                    ('G',[0,0,0]),
                    ('X',[1./2,1./2,0]),
                    ('W',[3./4,1./2,1./4]),
                    ('L',[1./2,1./2,1./2]),
                    ('G',[0,0,0]),
                    ('K',[3./4,3./8,3./8])                    
                    ]
        else:
            raise Exception("Bravais lattice name not found!")
            
        points=[x[1] for x in path]
        names=[x[0] for x in path]
        
        if nrpointspersegment > 1:
            path=self.point_path(points, nrpointspersegment)
            return points,names,path
        else:
            return points,names
        
    def __orbital_spreads_and_positions(self,wannier90_wout_filename):
        """
        Reads the final wannier90 orbital spreads and positions from the
        wannier90.wout file.
        
        wannier90_wout_filename: path to the wannier90.wout file.
        
        Return:
        spreads,positions
        """
        
        #TODO: What about performance? Stuff gets copied pretty often
        
        f = open(wannier90_wout_filename, 'r')
        lines = f.readlines()
        
        splitspaces = [[x.strip(',') for x in line.split()] for line in lines]
        #Catastrophe - only a Flatten[] command, but unreadable
        infile = [list(itertools.chain(*[x.split(',') for x in line])) for line in splitspaces]
        
        for nr,line in enumerate(lines):
            ret = line.find("Number of Wannier Functions")
            if ret >=0:
                break
            
        nrbands=int(infile[nr][6])
        
        start = infile.index(["Final","State"])
        data = infile[start+1:start+nrbands+1]
        
        f.close()
        
        spreads = [float(x[10]) for x in data]
        positions = [[float(x[6][:-1]),float(x[7][:-1]),float(x[8])] for x in data]
        
        return spreads,positions
    
    def __remove_duplicates(self,seq):
        #http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order
        #http://www.peterbe.com/plog/uniqifiers-benchmark (f2)
        checked = []
        for e in seq:
            if e not in checked:
                checked.append(e)
        return checked        

    
    def create_supercell_hamiltonian(self,cellcoordinates,latticevecs,usedhoppingcells='all',usedorbitals='all'):
        """
        Creates the matrix elements for a supercell containing several unit cells, e.g. a volume
        with one unit cell missing in the middle or one slice of a nanoribbon.
        
        cellcoordinates: unit cells contained in the supercell [[0,0,0],[1,0,0],...]. Use integergrid3d()
        to easily create a unit cell grid (which you can modify).
        latticevectors: lattice vectors of the new unit cell in the basis of the old unit cells (!!).
        E.g. a supercell of four graphene unit cells could have latticevecs=[[2,0,0],[0,2,0],[0,0,1]]. 
        If you want to create a ribbon or a molecule, use a high value in one of the coordinates
        (e.g. a long y lattice vector).
        
        Return:
        unitcellmatrixblocks: Hopping matrix elements, arranged by supercell.
        unitcellnumbers: Supercell coordinates of the blocks.
        newlatticevecs: New lattice vectors in real coordinates.
        usedhoppingcells: If you don't want to use all hopping parameters, you can set the cells to "hop"
        to here (list of cell coordinates).
        usedorbitals: a list of orbitals to use. Default is 'all'. Note: this only makes
        sense if the selected orbitals don't interact with other orbitals.       
        """
        
        #TODO: Naming (numbers,positions,coordinates) is ambiguous
        
        oldunitcellmatrixblocks=self.__unitcellmatrixblocks
        oldunitcellnumbers=self.__unitcellnumbers
        oldorbitalpositions=numpy.array(self.__orbitalpositions,copy=False)
        oldorbitalspreads=self.__orbitalspreads #Must be List, not numpy array!
        
        if usedhoppingcells == 'all':
            usedunitcellnrs=range(len(self.__unitcellnumbers))
        else:
            usedunitcellnrs=self.__unitcellcoordinates_to_nrs(usedhoppingcells)
        
        
        nr_unitcells_in_supercell=len(cellcoordinates)     
        
        unitcellmatrixblocks=[]
        unitcellnumbers=[]
        
        if usedorbitals=='all':
            orbitalnrs=range(self.__nrbands)
        else:
            orbitalnrs=usedorbitals
            
        orbitals_per_unitcell=len(orbitalnrs)
        
        #Set new orbital positions and spreads
        oldunitcellcoordinates=self.unitcellcoordinates(cellcoordinates)
        orbitalspreads=[oldorbitalspreads[i] for i in orbitalnrs]*nr_unitcells_in_supercell #Repeat oldorbitalspreads
        #print oldunitcellcoordinates
        #print oldorbitalpositions
        orbitalpositions=[list(oldorbitalpositions[orb]+cell) for cell in oldunitcellcoordinates for orb in orbitalnrs]
        
        metric_numerator,metric_denominator=self.__metric(latticevecs)
        
        #Loop over cells in supercell
        for cellnr,cell in enumerate(numpy.array(cellcoordinates)):
            #Loop over old blocks
            for i,oldnumber,oldblock in zip(range(len(oldunitcellmatrixblocks)),numpy.array(oldunitcellnumbers),oldunitcellmatrixblocks):
                if i in usedunitcellnrs:
                    oldblock_selectedorbitals=oldblock[numpy.ix_(orbitalnrs,orbitalnrs)]
                    hopto=cell+oldnumber
                    
                    hopto_scaled_times_metric_denominator=numpy.dot(numpy.dot(latticevecs,metric_numerator),hopto);
                    hopto_scaled=list(hopto_scaled_times_metric_denominator/metric_denominator)
                    hopto_rest_times_metric_denominator=hopto_scaled_times_metric_denominator%metric_denominator
                    hopto_nr=list(numpy.dot(hopto_rest_times_metric_denominator,latticevecs)/metric_denominator)
                    
                    try:
                        skip_block=False
                        hopto_nr_index=cellcoordinates.index(hopto_nr) 
                    except ValueError:
                        skip_block=True #if the cell to hop to is not in the cellcoordinates list, the block is skipped
                    
                    if skip_block==False:
                        #print hopto,hopto_scaled,hopto_nr,hopto_nr_index
    
                        """
                        #hopto_scaled=[x/y for x,y in zip(hopto,scaling)]
                        hopto_nr=[x%y for x,y in zip(hopto,scaling)]
                        hopto_nr_index=cellcoordinates.index(hopto_nr) #if the cell to hop to is not in the cellcoordinates list, this will fail -> then the block has to be skipped -> implement!!
                        """
                        #print hopfrom,oldnumber,hopto,hopto_scaled,hopto_nr,hopto_nr_index
                        
                        try:
                            unitcellindex=unitcellnumbers.index(hopto_scaled)
                        except ValueError:
                            unitcellindex=len(unitcellnumbers)
                            unitcellnumbers.append(hopto_scaled)
                            unitcellmatrixblocks.append(numpy.zeros((orbitals_per_unitcell*nr_unitcells_in_supercell,
                                                                    orbitals_per_unitcell*nr_unitcells_in_supercell),dtype=complex))
                        
                        #print unitcellindex    
                        unitcellmatrixblocks[unitcellindex][cellnr*orbitals_per_unitcell:(cellnr+1)*orbitals_per_unitcell,
                                                            hopto_nr_index*orbitals_per_unitcell:(hopto_nr_index+1)*orbitals_per_unitcell]=oldblock_selectedorbitals
                
        
        
        oldlatticevecs=self.__latticevecs.latticevecs()
        newlatticevecs=numpy.dot(numpy.array(latticevecs),oldlatticevecs) # (A.B)'=B'.A' - new latticevectors in real coordinates
        return self.from_raw_data(unitcellmatrixblocks, unitcellnumbers, newlatticevecs,orbitalspreads,orbitalpositions)
    
    def __metric(self,basis):
        """
        Calculates the metric for a given basis using the formula
        Inverse[basis].Inverse[Transpose[basis]].
        
        basis: 3x3 matrix, containing the basis vectors in rows.
        
        Return:        
        metric_numerator: The numerator of the metric (3x3 matrix)
        metric_denominator: The denominator - a scalar because it is identical for all elements.
        
        The metric is 1/metric_denominator * metric_numerator.
        """
        
        b=numpy.array(basis)
        
        metric_denominator=(-b[0,2]*b[1,1]*b[2,0]
                          +b[0,1]*b[1,2]*b[2,0]
                          +b[0,2]*b[1,0]*b[2,1]
                          -b[0,0]*b[1,2]*b[2,1]
                          -b[0,1]*b[1,0]*b[2,2]
                          +b[0,0]*b[1,1]*b[2,2])**2
        
        metric_numerator=[
                            [
                            (-(b[0][2]*b[1][1]) + b[0][1]*b[1][2])**2 + (b[0][2]*b[2][1] - b[0][1]*b[2][2])**2 + (-(b[1][2]*b[2][1]) + b[1][1]*b[2][2])**2,
                            (b[0][2]*b[1][0] - b[0][0]*b[1][2])*(-(b[0][2]*b[1][1]) + b[0][1]*b[1][2]) + (-(b[0][2]*b[2][0]) + b[0][0]*b[2][2])*(b[0][2]*b[2][1] - b[0][1]*b[2][2]) + (b[1][2]*b[2][0] - b[1][0]*b[2][2])*(-(b[1][2]*b[2][1]) + b[1][1]*b[2][2]),
                            (-(b[0][1]*b[1][0]) + b[0][0]*b[1][1])*(-(b[0][2]*b[1][1]) + b[0][1]*b[1][2]) + (b[0][1]*b[2][0] - b[0][0]*b[2][1])*(b[0][2]*b[2][1] - b[0][1]*b[2][2]) + (-(b[1][1]*b[2][0]) + b[1][0]*b[2][1])*(-(b[1][2]*b[2][1]) + b[1][1]*b[2][2])
                            ],[
                            (b[0][2]*b[1][0] - b[0][0]*b[1][2])*(-(b[0][2]*b[1][1]) + b[0][1]*b[1][2]) + (-(b[0][2]*b[2][0]) + b[0][0]*b[2][2])*(b[0][2]*b[2][1] - b[0][1]*b[2][2]) + (b[1][2]*b[2][0] - b[1][0]*b[2][2])*(-(b[1][2]*b[2][1]) + b[1][1]*b[2][2]),
                            (b[0][2]*b[1][0] - b[0][0]*b[1][2])**2 + (-(b[0][2]*b[2][0]) + b[0][0]*b[2][2])**2 + (b[1][2]*b[2][0] - b[1][0]*b[2][2])**2,
                            (-(b[0][1]*b[1][0]) + b[0][0]*b[1][1])*(b[0][2]*b[1][0] - b[0][0]*b[1][2]) + (b[0][1]*b[2][0] - b[0][0]*b[2][1])*(-(b[0][2]*b[2][0]) + b[0][0]*b[2][2]) + (-(b[1][1]*b[2][0]) + b[1][0]*b[2][1])*(b[1][2]*b[2][0] - b[1][0]*b[2][2])
                            ],[
                            (-(b[0][1]*b[1][0]) + b[0][0]*b[1][1])*(-(b[0][2]*b[1][1]) + b[0][1]*b[1][2]) + (b[0][1]*b[2][0] - b[0][0]*b[2][1])*(b[0][2]*b[2][1] - b[0][1]*b[2][2]) + (-(b[1][1]*b[2][0]) + b[1][0]*b[2][1])*(-(b[1][2]*b[2][1]) + b[1][1]*b[2][2]),
                            (-(b[0][1]*b[1][0]) + b[0][0]*b[1][1])*(b[0][2]*b[1][0] - b[0][0]*b[1][2]) + (b[0][1]*b[2][0] - b[0][0]*b[2][1])*(-(b[0][2]*b[2][0]) + b[0][0]*b[2][2]) + (-(b[1][1]*b[2][0]) + b[1][0]*b[2][1])*(b[1][2]*b[2][0] - b[1][0]*b[2][2]),
                            (-(b[0][1]*b[1][0]) + b[0][0]*b[1][1])**2 + (b[0][1]*b[2][0] - b[0][0]*b[2][1])**2 + (-(b[1][1]*b[2][0]) + b[1][0]*b[2][1])**2
                            ]
                            ]
        
        return metric_numerator,metric_denominator   
        
        
        
    
    def __absolute_ceiling(self,x):
        """
        Rounds to the next integer value that has the greater absolute value.
        
        1.5 -> 2
        -1.5 -> -2 
        """
        if x>=0:
            return int(numpy.ceil(x))
        else:
            return int(-numpy.ceil(-x)) 
    
    def create_modified_hamiltonian(self,usedhoppingcells='all',usedorbitals='all'):
        """
        Creates a Hamiltonian with dropped orbitals or hopping cells. This is just a wrapper for 
        create_supercell_hamiltonian().
        """
        
        return self.create_supercell_hamiltonian([[0,0,0]], [[1,0,0],[0,1,0],[0,0,1]], usedhoppingcells, usedorbitals)
    
    def integergrid3d(self,i,j,k):
        """
        Creates an integer grid with the dimensions i,j,k
        """
    
        return [[ii,jj,kk] for ii in range(i) for jj in range(j) for kk in range(k)]

class BandstructurePlot:
    """
    Combine several bandstructure plots.
    
    Call plot(kpoints,data) for every bandstructure plot.
    Then, call save(filename) to save to a file.
    """
    
    __stylelist=['b-','g-','r-','c-','m-','y-','k-']
    __plotcounter=0
    __myplot=0

    def __init__(self):
        self.__myplot=pyplot.figure()
        
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
    
    def set_aspect_ratio(self,aspect):
        """
        Set the aspect ratio. For possible values, see
        http://matplotlib.sourceforge.net/api/axes_api.html#matplotlib.axes.Axes.set_aspect
        """
        
        pyplot.figure(self.__myplot.number)
        ax = pyplot.gca()
        ax.set_aspect(aspect)
    
    def set_plot_range(self,**kwargs):
        """
        Set the plot range using the kwargs
        xmin, xmax, ymin, ymax.
        """

        #http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.axis
        
        pyplot.figure(self.__myplot.number)
        pyplot.axis(**kwargs)
        
        #ax = pyplot.gca()
        #ax.set_autoscale_on(False)
        
    
    def plot(self,kpoints,data,style='auto'):
        """
        Add a bandstructure plot to the figure.
        
        kpoints: list of kpoints
        data: list of eigenvalues for each kpoint.
        style: set format string of this plot. You can also set the 
        styles of all plots using setstyles().
        Default is 'auto', which means that the style list will be used.
        """
        
        """
        At the beginning, the figure is set to __myplot which was set in the constructor
        to avoid interference between plot functions.
        http://stackoverflow.com/questions/7986567/matplotlib-how-to-set-the-current-figure/7987462#7987462
        """
        pyplot.figure(self.__myplot.number) 

        pathlength=self.__kpoints_to_pathlength(kpoints)
        if style == 'auto':
            stylestring=self.__stylelist[self.__plotcounter % len(self.__stylelist)]
        else:
            stylestring=style
            
        self.__plotcounter+=1
        for band in data.transpose():
            pyplot.plot(pathlength,band,stylestring, linewidth=0.3)
            
    
    def save(self,filename):
        """
        Save the figure to a file. The format is determined
        by the filename.
        """
        pyplot.figure(self.__myplot.number) 
        pyplot.savefig(filename,dpi=(150))
        
    def reset(self):
        """
        Clear the current figure.
        """
        pyplot.figure(self.__myplot.number) 
        pyplot.clf()
        
    def show(self):
        pyplot.figure(self.__myplot.number) 
        pyplot.show()
    
    
