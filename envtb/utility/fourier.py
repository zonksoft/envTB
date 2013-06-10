import numpy
import envtb.quantumcapacitance.utilities as utilities

class FourierTransform:
    def __init__(self, latticevecs, data_on_grid, shape=None, axes=None):
        """
        supercell (int list): shape of supercell in multiples of latticevecs
        (as data_on_grid.shape would give it). The transformed data describes
        the supercell, the original data describes the original cell.
        latticevecs: The grid lattice vectors of data_on_grid.
        data_on_grid: 3D array, data grid.
        """
        self.original_data = data_on_grid
        self.original_latticevecs = latticevecs
        if shape is None:
            cell_shape = data_on_grid.shape
        else:
            cell_shape = shape
            
        if axes is None:
            axes = (0,1,2)
        
        self.reciprocal_latticevecs, self.transformed_grid = \
            FourierTransform.calculate_fourier_grid(self.original_latticevecs,
                                                    cell_shape)
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

    def fourier_transform(self, latticevecs, wave_functions):
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

            if len(wave_function.shape) == 2:
                transformed_orb_values_on_grid = orb.fourier_transform(wave_function.shape[:2], (0, 1))
                wannier_transformed = numpy.dstack((transformed_wave_function,)*transformed_orb_values_on_grid.shape[2]) * transformed_orb_values_on_grid.transformed_data
            elif len(wave_function.shape) == 3:
                transformed_orb_values_on_grid = orb.fourier_transform(wave_function.shape)
                wannier_transformed = transformed_wave_function * transformed_orb_values_on_grid.transformed_data
            else:
                raise ValueError('1D not supported')

            transformed_coefficients[orbnr] = wannier_transformed
            transformed_wave_functions[orbnr] = transformed_wave_function

        return transformed_coefficients, transformed_wave_functions
