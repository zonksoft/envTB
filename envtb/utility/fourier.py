import numpy
import envtb.quantumcapacitance.utilities as utilities
import envtb.wannier90.w90hamiltonian as w90
from envtb.wannier90.w90hamiltonian import Hamiltonian
import math
import matplotlib.pylab as plt

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
                wannier_transformed = self.__periodic_matrix(transformed_wave_function, *periodicity) * self.fourier_transformations[orbnr].transformed_data
            else:
                nx, ny, nz = transformed_wave_function.shape
                wannier_transformed = transformed_wave_function * self.fourier_transformations[orbnr].transformed_data[:nx, :ny, :nz]
            transformed_coefficients[orbnr] = wannier_transformed
            transformed_wave_functions[orbnr] = transformed_wave_function

        return transformed_coefficients, transformed_wave_functions
        
# XXX: Class is badly structured and documented      
class ZigzagGNRHelper:
    def __init__(self, nnfile, height, length, paddingx=0, paddingy=0, supercell_ham=True, ribbon_ham=True):
        """
        height: height of zGNR in 4-atom basiscells. MUST BE EVEN.
        length: number of slices. MUST BE EVEN.
        paddingx, paddingy: number of padding orbitals in x and y direction
        nnfile: Nearest neighbour file with geometry

        supercell_ham: should be set to False, to avoid calculation of calculate supercell_hamiltonian
        """
        print height, divmod(height, 2), length, divmod(length, 2)
        
        #if height % 2 != 0 or length % 2 != 0:
        #  raise ValueError('height and length must be even numbers.')
            
        self.height, self.length = height,length
        self.paddingx, self.paddingy = paddingx, paddingy
        self.nnfile = nnfile

        if supercell_ham: self.supercell_hamiltonian = self.__create_supercell_hamiltonian(nnfile, height, length)
        if ribbon_ham: 
            self.graphene_hamiltonian, self.doublecell_hamiltonian, self.ribbon_hamiltonian = \
                                        self.__create_hamiltonian(nnfile, height, length)
        else: 
            self.graphene_hamiltonian, self.doublecell_hamiltonian, self.ribbon_hamiltonian = \
                                        self.__create_hamiltonian(nnfile, height, length, rib_ham=False)

    @staticmethod
    def rings_to_atoms(nr_of_rings):
        return 2*nr_of_rings + 2

    @staticmethod
    def rings_to_2cells(nr_of_rings):
        """
        Returns the number of 2-cells (zigzag basis cell containing two
        graphene cells), including padding at top and bottom.
        """
        print nr_of_rings 
        if nr_of_rings % 2 == 0:
            return nr_of_rings/2 + 1
        else:
            return (nr_of_rings+1)/2 + 1

    @staticmethod
    def atoms_to_rings(nr_of_atoms):
        return (nr_of_atoms - 2)/2

    def __create_hamiltonian(self, nnfile, height, length, rib_ham=True):
        """
        Creates a graphene rectangle Hamiltonian from a nearest neighbour
        parameter file.

        nnfile: path to a nearest-neighbour parameter file.
        height: height of the rectangle in 4-atom unitcells (nr of atoms = 4*height*length)
        length: length of the rectangle (nr of stripes)

        
        Return:
        ham: Hamiltonian of the graphene unitcell.
        ham5: Hamiltonian of the graphene rectangle.
        if ham5 is not needed, to speed up the code use rib_ham=False
        """
       
        unitcells=height
        ham = Hamiltonian.from_nth_nn_list(nnfile)
        
        ham2 = ham.create_supercell_hamiltonian(
            [[0, 0, 0], [1, 0, 0]],
            [[1, -1, 0], [1, 1, 0], [0, 0, 1]])
        
        if rib_ham:
            ham3 = ham2.create_supercell_hamiltonian(
                    [[0, i, 0] for i in range(unitcells)],
                    [[1, 0, 0], [0, unitcells, 0], [0, 0, 1]])
        
            ham4 = ham3.create_modified_hamiltonian(
                    ham3.drop_dimension_from_cell_list(1))

            ham5 = ham4.create_supercell_hamiltonian([[i, 0, 0] for i in range(
                        length)], [[length, 0, 0], [0, 1, 0], [0, 0, 1]])
            return ham, ham2, ham5
        else: 
            return ham, ham2, ham2

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
        ham, ham2, ham5 = self.graphene_hamiltonian, self.doublecell_hamiltonian, self.ribbon_hamiltonian
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
        if self.height%2 == 0.0:
            splitvecs=numpy.split(vec, len(vec)/(4*self.height-2))
            splitvecs=[numpy.append(splitvec,0) for splitvec in splitvecs]
            splitvecs=[numpy.insert(splitvec,0,0) for splitvec in splitvecs]
        else:
            splitvecs=numpy.split(vec, len(vec)/(4*(self.height-1)))
            splitvecs=[numpy.append(splitvec,[0,0]) for splitvec in splitvecs]
            splitvecs=[numpy.insert(splitvec,[0,0],0) for splitvec in splitvecs]

        return numpy.array(splitvecs).flatten()

    def pad_vector(self, vec):
        splitvecs=numpy.split(vec,len(vec)/(4*self.height))
        splitvecs=[numpy.append(splitvec,[0]*self.paddingy) for splitvec in splitvecs]
        return numpy.concatenate((numpy.array(splitvecs).flatten(),numpy.zeros((self.paddingy+self.height)*self.paddingx)))

    def vector_to_grid(self, vec, height=None):
        if height is None:
            height = self.height*4
        return numpy.reshape(vec,(-1,height))


class GNRSimpleFourierTransform:
    def __init__(self, height_nr_atoms, length_nr_slices, nnfile):
        """
        height_nr_atoms: height of the zigzag ribbon in atoms.
        length_nr_slices: length of the ribbon in slices.
        nnfile: nearest-neighbour file which contains the graphene geometry.
        """
        self.height_nr_atoms = height_nr_atoms
        self.length_nr_slices = length_nr_slices
        self.zgh = self.__create_zigzag_gnr_helper(height_nr_atoms, length_nr_slices, nnfile)

        a=10
        self.localized_orbital_set = self.__create_gaussian_basis_orbitals(self.zgh, a)
        self.orbital_fourier_transform = RealSpaceWaveFunctionFourierTransform(
            self.localized_orbital_set,(self.zgh.length,self.zgh.height),(0,1))

        self.maxkx = self.orbital_fourier_transform.orbital_fourier_grid[-1][0][0][0]
        self.maxky = self.orbital_fourier_transform.orbital_fourier_grid[0][-1][0][1]


    def pad_and_split(self, wave_function):

        return self.__pad_and_split_zgnr_wavefunction(wave_function, self.zgh)

    def fourier_transform(self, wave_function):
        """
        Fourier transform a zigzag GNR wave function.

        Return:
        Fourier transform on a grid.

        """
        pad_and_split_wave_function = \
            self.__pad_and_split_zgnr_wavefunction(wave_function, self.zgh)

        transformed_coefficients, transformed_wave_functions = \
            self.orbital_fourier_transform.fourier_transform(pad_and_split_wave_function, full_transform=True)

        fourier_transform = self.__add_transformations(transformed_coefficients)
        fourier_wf = self.__add_transformations(transformed_wave_functions)

        return fourier_transform, fourier_wf
        
    def __create_zigzag_gnr_helper(self, height_nr_atoms, length_nr_slices, nnfile):
        """
        Create a ZigzagGNRHelper.
        height_nr_atoms: height of the zigzag ribbon in atoms.
        length_nr_slices: length of the ribbon in slices.
        nnfile: nearest-neighbour file which contains the graphene geometry.
        """
        height=ZigzagGNRHelper.rings_to_2cells(ZigzagGNRHelper.atoms_to_rings(height_nr_atoms))
        print height
        length=length_nr_slices 
        return ZigzagGNRHelper(nnfile, height, length, paddingx=0, paddingy=0, supercell_ham=False, ribbon_ham=False)

    def __pad_and_split_zgnr_wavefunction(self, wave_function, zgh):
        """
        Pad and split a zigzag-gnr wavefunction (geometry is given by zgr).
    
        Padding is necessary because a zigzag-ribbon is not periodic in y direction - some
        "ghost atoms" with no electron density are added at the top and the bottom.
    
        The wave function is split into four sublattice wave functions which correspond to the
        four atoms in the "graphene four-atom ribbon unit cell".
    
        wave_function: the wave function of the nanoribbon (down-up runs faster than left-right)
        zgh: ZigzagGNRHelper for the system.    
    
        Return:
        dictionary with the four sublattice wave functions.
        """
        padded = zgh.pad_vector(zgh.pad_nanoribbon_vector_to_periodic(wave_function))
        l1, l2, l3, l4 = [numpy.dstack((zgh.vector_to_grid(x, zgh.height),)) for x in zgh.split_sublattices(padded, 4)]
        pad_and_split_wave_function = {0: l1, 1: l2, 2: l3, 3:l4}

        return pad_and_split_wave_function

    def __create_gaussian_basis_orbitals(self, zgh, a):
        """
        Create a unit cell basis set with four basis orbitals (four-atom ribbon unit cell).
        One Gaussian is positioned on each atom.
    
        zgh: ZigzagGNRHelper for the system.
        a: Gaussian spread (arbitrary parameter)
    
        Return:
        LocalizedOrbitalSet containing the basis orbitals.
        """
        doublecell_ham = zgh.doublecell_hamiltonian
            
        unit_cell_grid = (2,2,1)
        gridpoints = 60
        
        def gaussian_orbital_at_center(x0, y0, z0):
            return w90.LocalizedOrbitalFromFunction(
                lambda x, y, z: (a/numpy.pi)**(3./2)*numpy.exp(-a*((x-x0)**2+(y-y0)**2+(z-z0)**2)),
                doublecell_ham.latticevectors()*unit_cell_grid, [-2,-1,0], gridpoints=gridpoints, dim2=True, 
                unit_cell_grid=unit_cell_grid)
        
        
        localized_orbital_set = w90.LocalizedOrbitalSet({i: gaussian_orbital_at_center(x0, y0, z0) for i,(x0,y0,z0) in 
                                            enumerate(doublecell_ham.orbitalpositions())}, doublecell_ham.latticevectors())
        return localized_orbital_set

    @staticmethod
    def roll_2d_center(data):
        return numpy.roll(numpy.roll(data,data.shape[0]/2,axis=0),data.shape[1]/2,axis=1)

    def __add_transformations(self, coeffs):
        """
        Add up a dictionary of arrays.
        It is used for adding up the contributions to the Fourier
        transformation from the different sublattices.

        coeffs: Dictionary containing numpy arrays.
        """
        transf_sum = numpy.zeros(coeffs.values()[0].shape, dtype=coeffs.values()[0].dtype)
        for transf in coeffs.values():
            transf_sum += transf
        return transf_sum

    @staticmethod
    def get_brillouin_zone():

        #Gamma_point = numpy.array([self.maxkx/2., self.maxky/2.])
        ag = 2.461 
        K_point_1 = 4. * numpy.pi / 3. / ag * numpy.array([1.0, 0.0])
        K_point_2 = 2. * numpy.pi / 3. / ag * numpy.array([-1.0, math.sqrt(3.0)])
        K_point_3 = 2. * numpy.pi / 3. / ag * numpy.array([-1.0, -math.sqrt(3.0)])
        Kp_point_1 = 2. * numpy.pi / 3. / ag * numpy.array([1.0, math.sqrt(3.0)])
        Kp_point_2 = 4. * numpy.pi / 3. / ag * numpy.array([-1.0, 0.0])
        Kp_point_3 = 2. * numpy.pi / 3. / ag * numpy.array([1.0, -math.sqrt(3.0)])
        Bril_zone = numpy.array([[K_point_1[0], Kp_point_1[0], K_point_2[0], Kp_point_2[0], K_point_3[0], Kp_point_3[0], K_point_1[0]],
                              [K_point_1[1], Kp_point_1[1], K_point_2[1], Kp_point_2[1], K_point_3[1], Kp_point_3[1], K_point_1[1]]])
        return Bril_zone


class GNRSimpleFastFourierTransform:
    def __init__(self, Nx=1, Ny=4, wave_function=None):
        
        self.Nx = Nx
        self.Ny = Ny
        if Ny%4 != 0:
            raise(ValueError, 'Ny should be devidable by 4')
        if wave_function is not None:
            self.wave_function = wave_function
            self.wave_funcion_fourier, self.wave_function_fourier_sublattices = self.make_fourier()

    def make_fourier(self, wave_function=None):
           #print wf_arr
        if wave_function is not None:
            self.wave_function=wave_function
        wf_arr_split = numpy.split(self.wave_function, self.Nx)
        wf_arr_split = [numpy.append(splitvec, [0, 0]) for splitvec in wf_arr_split]
        wf_arr_split = [numpy.insert(splitvec, [0, 0], 0) for splitvec in wf_arr_split]
        wf_arr_fl = numpy.array(wf_arr_split).flatten()
        wf_fl = {i: numpy.array(wf_arr_fl[i::4]).reshape(self.Nx, self.Ny/4+1) for i in xrange(4)}
        wf_four = {i: numpy.fft.fft2(wf_fl[i]) for i in xrange(4)}
        wf_all = wf_four[0] +  wf_four[1] + wf_four[2] + wf_four[3]

        self.wave_function_fourier = wf_all
        self.wave_function_fourier_sublattices = wf_four
        return wf_all, wf_four

    @staticmethod
    def plot_fourier_transform(wave_function_fourier, N=5, figuresize = (14,14)):
        a = 1.42
        wf_per = numpy.dstack((numpy.vstack((numpy.hstack((wave_function_fourier,) * N),) * N),) * 1)

        BZ = GNRSimpleFourierTransform.get_brillouin_zone()

        plt.figure(figsize=figuresize)

        plt.imshow(abs(wf_per[:,:,0]).transpose(), interpolation='nearest', aspect=1.0, 
                    extent=[0.0, numpy.sqrt(3)*2.*N*numpy.pi/3./a, 0.0, 2.*N*numpy.pi/3./a])
        plot_indexes = [[nx, ny] for nx in xrange(0, 2*(N+1), 2) for ny in xrange(0, 2*(N+1), 2) if (nx-ny)%4 == 0]
        [plt.plot(BZ[0]+numpy.sqrt(3)*nx*numpy.pi/3./a, BZ[1]+ny*numpy.pi/3./a, 'w') for nx,ny in plot_indexes]

        plt.plot([0.0, numpy.sqrt(3)*2.*numpy.pi/3./a, numpy.sqrt(3)*2.*numpy.pi/3./a], 
            [2.*numpy.pi/3./a, 2.*numpy.pi/3./a, 0.0], 'w', ls='--')

        plt.xlim(0.0, numpy.sqrt(3)*2.*N*numpy.pi/3./a)
        plt.ylim(0.0, 2.*N*numpy.pi/3./a)
        return None


class GNRSimpleFastInvertFourierTransform:

    def __init__(self, Nx=1, Ny=1, wave_function_fourier=None, repeat=False):
        if wave_function_fourier.__class__ is dict:
            self.wave_function_fourier = wave_function_fourier
        else:
            if repeat:
                self.wave_function_fourier ={i: wave_function_fourier for i in xrange(4)}
            else:
                self.wave_function_fourier = {i: numpy.zeros((Nx, Ny/4+1), dtype = complex)  for i in xrange(4)}
                if wave_function_fourier is not None:
                    self.wave_function_fourier[0] = wave_function_fourier
        self.Nx = Nx
        self.Ny = Ny
        self.wave_function = self.inverse_fourier_transform()

    def __remove_cites(self, inverse_fourier):
        wf_arr_i_split = numpy.split(inverse_fourier, self.Nx)
        wf_arr_i_split = [splitvec[2:-2] for splitvec in wf_arr_i_split]
        return numpy.array(wf_arr_i_split).flatten()

    def inverse_fourier_transform(self):
        wf_i = {i: numpy.fft.ifft2(self.wave_function_fourier[i]).flatten() for i in xrange(4)}
        wf_arr_i = numpy.zeros(4*len(wf_i[0]), dtype=complex)
        wf_arr_i[::4] = wf_i[0]
        wf_arr_i[1::4] = wf_i[1]
        wf_arr_i[2::4] = wf_i[2]
        wf_arr_i[3::4] = wf_i[3]
        wf_arr_i_split = self.__remove_cites(wf_arr_i)
        return wf_arr_i_split
