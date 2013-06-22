import numpy
import envtb.quantumcapacitance.utilities as utilities
from envtb.wannier90.w90hamiltonian import Hamiltonian

class FourierTransform:
    def __init__(self, latticevecs, data_on_grid, shape=None, axes=None):
        """
        supercell (int list): shape of supercell in multiples of latticevecs
        (as data_on_grid.shape would give it). The transformed data describes
        the supercell, the original data describes the original cell.
        latticevecs: The grid lattice vectors of data_on_grid.
        data_on_grid: 3D array, data grid.
        shape: shape of the wave function coefficients.
        """
        self.original_data = data_on_grid
        self.original_latticevecs = latticevecs
        if shape is None:
            cell_shape = data_on_grid.shape
        else:
            cell_shape = [x*y for x, y in zip(shape, data_on_grid.shape)]
            
        if axes is None:
            axes = (0,1,2)
        
        #self.reciprocal_latticevecs, self.transformed_grid = \
        #    FourierTransform.calculate_fourier_grid(self.original_latticevecs,
        #                                            cell_shape)
        self.transformed_data = self.__calculate_fourier_transform(self.original_data, cell_shape, axes)
        
        
    """
    def __fill_up_with_zeros(self, data, new_shape):
        old_shape = data.shape
        
        b=numpy.vstack((data, numpy.zeros([new_shape[0]-old_shape[0],old_shape[1]             ,old_shape[2]             ])))
        c=numpy.hstack((b,    numpy.zeros([new_shape[0]             ,new_shape[1]-old_shape[1],old_shape[2]             ])))
        d=numpy.dstack((c,    numpy.zeros([new_shape[0]             ,new_shape[1]             ,new_shape[2]-old_shape[2]])))
        
        return d
    """
        
    def __calculate_fourier_transform(self, original_data, shape, axes):
        """
        shape: if not None, original_data will be filled up with zeroes
        to the given shape and then transformed.
        """
        return numpy.fft.fftn(original_data, s=shape, axes=axes)

    @staticmethod
    def calculate_fourier_grid(latticevecs, data_shape):
        reciprocal_latticevecs = numpy.array([numpy.cross(latticevecs[1], latticevecs[2]),
                                               numpy.cross(latticevecs[2], latticevecs[0]),
                                               numpy.cross(latticevecs[0], latticevecs[1])
                                               ])
        reciprocal_latticevecs *= 2 * numpy.pi / numpy.linalg.det(latticevecs)

        gridsize = data_shape

        return reciprocal_latticevecs / data_shape, \
            numpy.array([[[numpy.dot([i, j, k], reciprocal_latticevecs)
                  for k in numpy.arange(0, 1, 1. / gridsize[2])]
                 for j in numpy.arange(0, 1, 1. / gridsize[1])]
                for i in numpy.arange(0, 1, 1. / gridsize[0])])

    def plot(self, ax):
        data_to_plot = None
        im = ax.imshow(data_to_plot)
        return im


class RealSpaceWaveFunctionFourierTransform:
    def __init__(self, real_space_orbitals):
        """
        real_space_orbitals: needs orbitals property which is a list of orbitals
        """
        self.fourier_transformations = {}
        self.wannier_real_space_orbitals = real_space_orbitals

    # XXX: write function to go from wave function in vector to array

    def __periodic_matrix(self, matrix, ni, nj, nk):
        """
        ni: row multiplicator
        nj: column multiplicator
        nk: slice multiplicator
        """

        return numpy.dstack((numpy.vstack((numpy.hstack((matrix,)*nj),)*ni),)*nk)

    def fourier_transform(self, latticevecs, wave_functions, axes=(0,1,2)):
        """
        wave_functions: Coefficients of the wave function on a 
        nonorthogonal grid (in a multidimensional array),
        one array per orbital in a dictionary which the
        orbital number as key.
        latticevecs: grid that spans the unit lattice, as an integer
        multiple of unit cells.

        If wave_functions contains 2D arrays, an xy plane is assumed and
        the z coordinate of the real space orbitals is not transformed.
        """

        global transformed_wave_function
        global transformed_orb_values_on_grid

        transformed_coefficients = {}
        transformed_wave_functions = {}
        for orbnr, orb in \
            self.wannier_real_space_orbitals.orbitals.iteritems():
            # XXX: if you transform several wave functions, you can save this

            wave_function = numpy.array(wave_functions[orbnr], copy=False)
            transformed_wave_function = numpy.fft.fftn(wave_function)
            transformed_lattice_vectors, transformed_grid = \
                FourierTransform.calculate_fourier_grid(
                    latticevecs, wave_function.shape)

            transformed_orb_values_on_grid = orb.fourier_transform(wave_function.shape, axes)
            periodicity = [i/j for i,j in zip(transformed_orb_values_on_grid.transformed_data.shape, transformed_wave_function.shape)]
            print periodicity
            wannier_transformed = self.__periodic_matrix(transformed_wave_function,*periodicity) * transformed_orb_values_on_grid.transformed_data

            transformed_coefficients[orbnr] = wannier_transformed
            transformed_wave_functions[orbnr] = transformed_wave_function

        return transformed_coefficients, transformed_wave_functions
        
# XXX: Class is badly structured and documented      
class ZigzagGNRHelper:
    def __init__(self, nnfile, height, length, paddingx=0, paddingy=0):
        """
        height: height of zGNR in 4-atom basiscells
        length: number of slices
        paddingx, paddingy: number of padding orbitals in x and y direction
        nnfile: Nearest neighbour file with geometry
        """
        self.height, self.length = height,length
        self.paddingx, self.paddingy = paddingx, paddingy
        self.nnfile = nnfile
        
    def __create_hamiltonian(self, nnfile, height, length):
    
        unitcells=self.height/2
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
        
        return ham, ham5


    def __create_transformations(self, ham):
        a1, a2, a3 = ham.latticevectors()
        b1 = 8*a1
        b2 = 4*a2
        b3 = a3
        
        c1=b1
        c2=(a1-a2)*4
        c3=a3

        vecs=numpy.vstack((b1,b2,b3))
        gr_vecs=numpy.vstack((a1,a2,a3))
        invvecs = numpy.linalg.inv(vecs)
        invgr_vecs = numpy.linalg.inv(gr_vecs)
        
        return vecs, gr_vecs, invvecs, invgr_vecs, a1, a2, a3, b1, b2, b3, c1, c2, c3

    def split_sublattices(self, vec, nrorbs):
        return numpy.array([[vec[i] for i in range(start,len(vec),nrorbs)] for start in range(nrorbs)])

    def __shift_vector(self, vec, b1, b2, c2, invvecs):
        transformed = numpy.dot(vec,invvecs)
        transformed_right = numpy.dot(vec-c2,invvecs)
        if transformed[1] < 0:
            if transformed_right[0] < 0:
                return vec + b1/2 + b2
            else:
                return vec - b1/2 + b2
        else:
            return vec
        
    def resort_nanoribbon(self):    
        ham, ham5 = self.__create_hamiltonian(self.nnfile, self.height, self.length)
        vecs, gr_vecs, invvecs, invgr_vecs, a1, a2, a3, b1, b2, b3, c1, c2, c3 = self.__create_transformations(ham)
          
        orbpos = numpy.array(ham5.orbitalpositions())
        
        coords = self.split_sublattices(orbpos,2)[0]
        shiftvecs = numpy.array([self.__shift_vector(vec, b1, b2, c2, invvecs) for vec in coords])
        enumeration=numpy.array([range(len(shiftvecs))]).transpose()
        l=numpy.round(numpy.hstack((numpy.dot(shiftvecs,invgr_vecs),enumeration))*3)/3
        ind=numpy.lexsort((l[:,0],l[:,1]))    
        return l[ind][:,3]

    def pad_nanoribbon_vector_to_periodic(self, vec):
        splitvecs=numpy.split(vec,len(vec)/(4*self.height-2))
        splitvecs=[numpy.append(splitvec,0) for splitvec in splitvecs]
        splitvecs=[numpy.insert(splitvec,0,0) for splitvec in splitvecs]
        return numpy.array(splitvecs).flatten()

    def pad_vector(self, vec):
        splitvecs=numpy.split(vec,len(vec)/(4*self.height))
        splitvecs=[numpy.append(splitvec,[0]*self.paddingy) for splitvec in splitvecs]
        return numpy.concatenate((numpy.array(splitvecs).flatten(),numpy.zeros((self.paddingy+self.height)*self.paddingx)))

    def vector_to_grid(self, vec):
        return numpy.reshape(vec,(-1,self.height*4))
