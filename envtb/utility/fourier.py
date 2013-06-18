import numpy

class FourierTransform:
    def __init__(self, latticevecs, data_on_grid, shape=None, axes=None):
        """
        latticevecs: The grid lattice vectors of data_on_grid.
        data_on_grid: 3D array, data grid.
        shape: shape of the supercell (3-tuple), which is empty except for the
        given data grid, in integer multiples of the data grid size.
        Default is None, which equals (1,1,1).
        axes: Axes along which the FT will be performed (Default None,
        which equals (0,1,2) ), e.g. (0,1) if the z axis shall not be
        transformed.
        """
        self.original_data = data_on_grid
        self.original_latticevecs = latticevecs
        if shape is None:
            cell_shape = data_on_grid.shape
        else:
            cell_shape = [x * y for x, y in zip(shape, data_on_grid.shape)]

        if axes is None:
            axes = (0, 1, 2)
        else:
            cell_shape = [j for i, j in enumerate(shape) if i in axes]

        # self.reciprocal_latticevecs, self.transformed_grid = \
        #    FourierTransform.calculate_fourier_grid(self.original_latticevecs,
        #                                            cell_shape)
        self.transformed_data = self.__calculate_fourier_transform(self.original_data, cell_shape, axes)

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
        fourier_grid_latticevecs: lattice vectors of the Fourier grid.
        fourier_grid: the grid itself.
        """
        reciprocal_latticevecs = numpy.array([numpy.cross(latticevecs[1], latticevecs[2]),
                                               numpy.cross(latticevecs[2], latticevecs[0]),
                                               numpy.cross(latticevecs[0], latticevecs[1])
                                               ])
        reciprocal_latticevecs *= 2 * numpy.pi / numpy.linalg.det(latticevecs)

        gridsize = data_shape

        fourier_grid_latticevecs = reciprocal_latticevecs / data_shape

        fourier_grid = numpy.array([[[numpy.dot([i, j, k], reciprocal_latticevecs)
                  for k in numpy.arange(0, 1, 1. / gridsize[2])]
                 for j in numpy.arange(0, 1, 1. / gridsize[1])]
                for i in numpy.arange(0, 1, 1. / gridsize[0])])

        return fourier_grid_latticevecs, fourier_grid


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
        wave_function_shape: Shape of the wave functions which will be transformed.
        transform_axes: Axes which will be transformed; e.g. (0,1) means that the
        z axis will not be transformed.

        Usage example:
        >>> latticevecs = numpy.eye(3)
        >>> wave_function = numpy.dstack((numpy.random.random((10,10))-0.5,))
        >>> transformed_coefficients, transformed_wave_functions = \
        ... real_space_wave_function_fourier_transform.fourier_transform(
        ... latticevecs, {1: wave_function})
        """
        self.fourier_transformations = {}
        self.wannier_real_space_orbitals = real_space_orbitals

        for orbnr, orb in \
            self.wannier_real_space_orbitals.orbitals.iteritems():
            self.fourier_transformations[orbnr] = orb.fourier_transform(wave_function_shape)


    # XXX: write function to go from wave function in vector to array

    def __periodic_matrix(self, matrix, ni, nj, nk):
        """
        ni: row multiplicator
        nj: column multiplicator
        nk: slice multiplicator
        """

        return numpy.dstack((numpy.vstack((numpy.hstack((matrix,) * nj),) * ni),) * nk)

    def fourier_transform(self, latticevecs, wave_functions, full_transform=False):
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
        latticevecs: grid that spans the unit lattice, as an integer
        multiple of unit cells.
        full_transform: If True, the output will contain all wavelengths; if False,
        only the wavelengths which occur on the wave function lattice, not those
        in the real space orbital lattice.

        Return:
        transformed_coefficients: the real space fourier transforms (dict)
        transformed_wave_functions: the Fourier transforms of the lattice WF (dict)
        """

        transformed_coefficients = {}
        transformed_wave_functions = {}
        for orbnr, orb in \
            self.wannier_real_space_orbitals.orbitals.iteritems():
            # XXX: if you transform several wave functions, you can save this

            wave_function = numpy.array(wave_functions[orbnr], copy=False)
            transformed_wave_function = numpy.fft.fftn(wave_function)
            # transformed_lattice_vectors, transformed_grid = \
            #    FourierTransform.calculate_fourier_grid(
            #        latticevecs, wave_function.shape)

            transformed_orb_values_on_grid = orb.fourier_transform(wave_function.shape)
            if full_transform:
                periodicity = transformed_orb_values_on_grid.original_data.shape
                wannier_transformed = self.__periodic_matrix(transformed_wave_function, *periodicity) * self.fourier_transformations[orbnr].transformed_data
            else:
                nx, ny, nz = transformed_wave_function.shape
                wannier_transformed = transformed_wave_function * self.fourier_transformations[orbnr].transformed_data[:nx, :ny, :nz]
            transformed_coefficients[orbnr] = wannier_transformed
            transformed_wave_functions[orbnr] = transformed_wave_function

        return transformed_coefficients, transformed_wave_functions
