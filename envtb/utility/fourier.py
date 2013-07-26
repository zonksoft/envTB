import numpy
import envtb.quantumcapacitance.utilities as utilities
from envtb.wannier90.w90hamiltonian import Hamiltonian

class FourierTransform:
    def __init__(self, latticevecs, data_on_grid, shape=None, axes=None):
        """
        latticevecs: The grid lattice vectors of data_on_grid.
        data_on_grid: 3D array, data grid.
        shape: shape of the supercell (3-tuple), which is empty except for the
        given data grid, in gridpoints.
        Default is None, in which case the shape of the data will be used.
        axes: Axes along which the FT will be performed (Default None,
        which equals (0,1,2) ), e.g. (0,1) if the z axis shall not be
        transformed.
        """
        self.original_data = data_on_grid
        self.original_latticevecs = latticevecs
        
        
        
        self.reshaped_latticevecs = self.__reshape_latticevecs(latticevecs, shape, data_on_grid)
        
        print self.original_latticevecs
        print self.reshaped_latticevecs
        
        if shape is None:
            cell_shape = data_on_grid.shape
        else:
            cell_shape = shape

        if axes is None:
            axes = (0, 1, 2)
        #else:
        #    cell_shape = [j for i, j in enumerate(shape) if i in axes]

        self.reciprocal_latticevecs, self.transformed_grid = \
           FourierTransform.calculate_fourier_grid(self.reshaped_latticevecs,
                                                   cell_shape)
        self.transformed_data = self.__calculate_fourier_transform(self.original_data, cell_shape, axes)


    def __reshape_latticevecs(self, latticevecs, data_shape, data_on_grid):
        if len(data_shape) == 1:
            data_shape = numpy.array(list(data_shape) + [1,1])
            
        elif len(data_shape) == 2:
            data_shape = numpy.array(list(data_shape) + [1])            
        else:
            data_shape = numpy.array(data_shape)
            
        return latticevecs * (numpy.array(data_shape, dtype=numpy.float) / numpy.array(data_on_grid.shape))
            
    def __calculate_fourier_transform(self, original_data, shape, axes):
        """
        shape: if not None, original_data will be filled up with zeroes
        to the given shape and then transformed.
        axes: Axes to transform.
        """
        return numpy.fft.fftn(original_data, s=shape, axes=axes)

    @staticmethod
    def calculate_fourier_grid(latticevecs, data_shape):
        """
        Calculate the reciprocal Fourier grid for given lattice vectors and
        data shape.

        Return:
        reciprocal_latticevecs: reciprocal lattice vectors.
        fourier_grid_latticevecs: lattice vectors of the Fourier grid.
        fourier_grid: the grid itself.
        """
        reciprocal_latticevecs = numpy.array([numpy.cross(latticevecs[1], latticevecs[2]),
                                               numpy.cross(latticevecs[2], latticevecs[0]),
                                               numpy.cross(latticevecs[0], latticevecs[1])
                                               ])
        reciprocal_latticevecs *= 2 * numpy.pi / numpy.linalg.det(latticevecs)


        
        if len(data_shape) == 1:
            data_shape = numpy.array(list(data_shape) + [1,1])
            
        elif len(data_shape) == 2:
            data_shape = numpy.array(list(data_shape) + [1])            
        else:
            data_shape = numpy.array(data_shape)

        #fourier_grid_latticevecs = reciprocal_latticevecs * data_shape
        
        gridsize = data_shape        

        fourier_grid = numpy.array([[[numpy.dot([i, j, k], reciprocal_latticevecs)
                  for k in numpy.arange(0, gridsize[2])]
                 for j in numpy.arange(0, gridsize[1])]
                for i in numpy.arange(0, gridsize[0])])

        return reciprocal_latticevecs, fourier_grid


    def plot(self, ax, kzidx=0, value='abs'):
        """
        Plot the Fourier transformation.

        ax: Axes to plot to.
        kzidx: index of the kz plane to plot (default 0)
        value: 'abs', 're' or 'im'
        """
        if value == 'abs':
            data_to_plot = abs(self.transformed_data[:, :, kzidx])
        if value == 're':
            data_to_plot = self.transformed_data[:, :, kzidx].real
        if value == 'im':
            data_to_plot = self.transformed_data[:, :, kzidx].imag
        im = ax.imshow(data_to_plot, interpolation='nearest', origin='bottom')
        return im


class RealSpaceWaveFunctionFourierTransform:
    def __init__(self, real_space_orbitals, wave_function_shape, transform_axes=(0, 1, 2)):
        """
        Handles the Fourier transform of tight-binding wave functions using a
        real-space basis set.

        The Fourier transforms of the real-space basis orbitals are calculated
        at the instantiation.

        real_space_orbitals: Orbital set (instance of w90hamiltonian.LocalizedOrbitalSet)
        wave_function_shape: Shape of the wave functions which will be transformed, in multiples
        of the unit cell.
        transform_axes: Axes which will be transformed; e.g. (0,1) means that the
        z axis will not be transformed.

        Usage example:
        >>> latticevecs = numpy.eye(3)
        >>> wave_function = numpy.dstack((numpy.random.random((10,10))-0.5,))
        >>> transformed_coefficients, transformed_wave_functions = \
        ... real_space_wave_function_fourier_transform.fourier_transform(
        ... latticevecs, {1: wave_function})
        """
        
        # XXX stuff is called wannier_real_space_orbitals, but should be called 
        # real_space_orbitals because they can be any instance of LocalizedOrbitalSet
        
        self.fourier_transformations = {}
        self.wannier_real_space_orbitals = real_space_orbitals
        
        self.lattice_reciprocal_latticevecs, self.lattice_fourier_grid = \
            FourierTransform.calculate_fourier_grid(
                real_space_orbitals.latticevecs*(self.__fill_shape_to_3d(wave_function_shape)), wave_function_shape)
               

        for orbnr, orb in \
            self.wannier_real_space_orbitals.orbitals.iteritems():
            self.fourier_transformations[orbnr] = orb.fourier_transform(wave_function_shape, axes=transform_axes)
            
        self.orbital_reciprocal_latticevecs, self.orbital_fourier_grid = \
            self.fourier_transformations.values()[0].reciprocal_latticevecs, \
            self.fourier_transformations.values()[0].transformed_grid            

    def __fill_shape_to_3d(self, shape):
        if len(shape) == 1:
            filled_shape = numpy.array(list(shape) + [1,1])
            
        elif len(shape) == 2:
            filled_shape = numpy.array(list(shape) + [1])            
        else:
            filled_shape = numpy.array(shape)
            
        return filled_shape
        
    # XXX: write function to go from wave function in vector to array

    def __periodic_matrix(self, matrix, ni, nj, nk):
        """
        ni: row multiplicator
        nj: column multiplicator
        nk: slice multiplicator
        """

        return numpy.dstack((numpy.vstack((numpy.hstack((matrix,) * nj),) * ni),) * nk)

    def fourier_transform(self, wave_functions, full_transform=False):
        """
        Fourier transform a given wave function.

        wave_functions: Coefficients of the wave function on a 
        nonorthogonal grid (in a multidimensional array),
        one array per orbital in a dictionary which the
        orbital number as key. In order to be representable by an array,
        the wave function needs to be periodic and a cuboid, in terms of its
        lattice vectors. Every periodic WF which is not a cuboid can be transformed
        to a cuboid by periodic rearrangement.
        The WF needs to have the shape previously supplied to the constructor.
        full_transform: If True, the output will contain all wavelengths; if False,
        only the wavelengths which occur on the wave function lattice, not those
        in the real space orbital lattice.

        Return:
        transformed_coefficients: the real space fourier transforms (dict)
        transformed_wave_functions: the Fourier transforms of the lattice WF (dict)
        """

        # XXX: full_transform = True gives only positive frequencies, what about the
        # negative ones?
        
        transformed_coefficients = {}
        transformed_wave_functions = {}
        for orbnr, orb in \
            self.wannier_real_space_orbitals.orbitals.iteritems():
            # XXX: if you transform several wave functions, you can save this

            wave_function = numpy.array(wave_functions[orbnr], copy=False)
            transformed_wave_function = numpy.fft.fftn(wave_function)


            if full_transform:
                periodicity = numpy.array(self.fourier_transformations[orbnr].transformed_data.shape) / numpy.array(transformed_wave_function.shape)
                print periodicity
                wannier_transformed = self.__periodic_matrix(transformed_wave_function, *periodicity) * self.fourier_transformations[orbnr].transformed_data
            else:
                nx, ny, nz = transformed_wave_function.shape
                wannier_transformed = transformed_wave_function * self.fourier_transformations[orbnr].transformed_data[:nx, :ny, :nz]
            transformed_coefficients[orbnr] = wannier_transformed
            transformed_wave_functions[orbnr] = transformed_wave_function

        return transformed_coefficients, transformed_wave_functions
        
# XXX: Class is badly structured and documented      
class ZigzagGNRHelper:
    def __init__(self, nnfile, height, length, paddingx=0, paddingy=0):
        """
        height: height of zGNR in 4-atom basiscells. MUST BE EVEN.
        length: number of slices. MUST BE EVEN.
        paddingx, paddingy: number of padding orbitals in x and y direction
        nnfile: Nearest neighbour file with geometry
        """
        
        if height % 2 != 0 or length % 2 != 0:
            raise ValueError('height and length must be even numbers.')
            
        self.height, self.length = height,length
        self.paddingx, self.paddingy = paddingx, paddingy
        self.nnfile = nnfile
        
        self.supercell_hamiltonian = self.__create_supercell_hamiltonian(nnfile, height, length)
        
        
    @staticmethod
    def rings_to_atoms(nr_of_rings):
        return 2*nr_of_rings + 2
        
    @staticmethod
    def rings_to_2cells(nr_of_rings):
        """
        Returns the number of 2-cells (zigzag basis cell containing two
        graphene cells), including padding at top and bottom.
        """
        
        if nr_of_rings % 2 == 0:
            return nr_of_rings/2 + 1
        else:
            return (nr_of_rings+1)/2 + 1   
            
    @staticmethod        
    def atoms_to_rings(nr_of_atoms):
        return (nr_of_atoms - 2)/2                 
        
    def create_hamiltonian(self, nnfile, height, length):
        """
        Creates a graphene rectangle Hamiltonian from a nearest neighbour
        parameter file.

        nnfile: path to a nearest-neighbour parameter file.
        height: height of the rectangle in 4-atom unitcells (nr of atoms = 4*height*length)
        length: length of the rectangle (nr of stripes)

        
        Return:
        ham: Hamiltonian of the graphene unitcell.
        ham5: Hamiltonian of the graphene rectangle.
        """
       
        unitcells=height
        ham = Hamiltonian.from_nth_nn_list(nnfile)
        
        ham2 = ham.create_supercell_hamiltonian(
            [[0, 0, 0], [1, 0, 0]],
            [[1, -1, 0], [1, 1, 0], [0, 0, 1]])
        
        ham3 = ham2.create_supercell_hamiltonian(
            [[0, i, 0] for i in range(unitcells)],
            [[1, 0, 0], [0, unitcells, 0], [0, 0, 1]])
        
        ham4 = ham3.create_modified_hamiltonian(
            ham3.drop_dimension_from_cell_list(1))
        
        ham5 = ham4.create_supercell_hamiltonian([[i, 0, 0] for i in range(
            length)], [[length, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        return ham, ham2, ham5
        
    def __create_supercell_hamiltonian(self, nnfile, height, length):
        unitcells=length*2
        ham = Hamiltonian.from_nth_nn_list(nnfile)
        
        ham3 = ham.create_supercell_hamiltonian(
            [[i, j, 0] for j in range(height) for i in range(unitcells)],
            [[unitcells, 0, 0], [0, height, 0], [0, 0, 1]])
        
        #ham4 = ham3.create_modified_hamiltonian(
        #    ham3.drop_dimension_from_cell_list(1))
        
        
        return ham3


    def __create_transformations(self, ham):
        a1, a2, a3 = ham.latticevectors()
        b1 = 2*self.length*a1
        b2 = self.height*a2
        b3 = a3
        
        c1=b1
        c2=(a1+a2)*self.height
        c3=a3

        vecs=numpy.vstack((b1,b2,b3))
        gr_vecs=numpy.vstack((a1,a2,a3))
        lattice_vecs =numpy.vstack((c1, c2, c3))
        invvecs = numpy.linalg.inv(vecs)
        invgr_vecs = numpy.linalg.inv(gr_vecs)
        invlattice_vecs = numpy.linalg.inv(lattice_vecs)
        
        return vecs, gr_vecs, lattice_vecs, invvecs, invgr_vecs, invlattice_vecs

    def split_sublattices(self, vec, nrorbs):
        return numpy.array([[vec[i] for i in range(start,len(vec),nrorbs)] for start in range(nrorbs)])

    def __shift_vectors(self, coords, vecs, invvecs, lattice_vecs, invlattice_vecs):
        
        coords2 = coords - numpy.dot(numpy.floor(numpy.dot(coords, invlattice_vecs)), lattice_vecs)
        coords3 = coords2 - numpy.dot(numpy.floor(numpy.dot(coords2, invvecs)), vecs)
        
        return coords3
        
    def resort_nanoribbon(self):    
        ham, ham2, ham5 = self.create_hamiltonian(self.nnfile, self.height, self.length)
        vecs, gr_vecs, lattice_vecs, invvecs, invgr_vecs, invlattice_vecs, = self.__create_transformations(ham)
          
        orbpos = numpy.array(ham5.orbitalpositions())
        
        coords = self.split_sublattices(orbpos,2)[0]
        shiftvecs = self.__shift_vectors(coords, vecs, invvecs, lattice_vecs, invlattice_vecs)
        enumeration=numpy.array([range(len(shiftvecs))]).transpose()
        l=numpy.round(numpy.hstack((numpy.dot(shiftvecs,invgr_vecs),enumeration))*3)/3
        ind=numpy.lexsort((l[:,0],l[:,1]))    
        return l[ind][:,3]
        
    def resort_nanoribbon_vector(self, vec):
        return numpy.array([vec[2*int(i)+j] for i in self.resort_nanoribbon() for j in range(2)])
                
    def pad_nanoribbon_vector_to_periodic(self, vec):
        splitvecs=numpy.split(vec,len(vec)/(4*self.height-2))
        splitvecs=[numpy.append(splitvec,0) for splitvec in splitvecs]
        splitvecs=[numpy.insert(splitvec,0,0) for splitvec in splitvecs]
        return numpy.array(splitvecs).flatten()

    def pad_vector(self, vec):
        splitvecs=numpy.split(vec,len(vec)/(4*self.height))
        splitvecs=[numpy.append(splitvec,[0]*self.paddingy) for splitvec in splitvecs]
        return numpy.concatenate((numpy.array(splitvecs).flatten(),numpy.zeros((self.paddingy+self.height)*self.paddingx)))

    def vector_to_grid(self, vec, height=None):
        if height is None:
            height = self.height*4
        return numpy.reshape(vec,(-1,height))
