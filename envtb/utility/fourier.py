import numpy
import envtb.quantumcapacitance.utilities as utilities

class FourierTransform:
    def __init__(self, latticevecs, data_on_grid):
        self.original_data = data_on_grid
        self.original_latticevecs = latticevecs
        reciprocal_latticevecs, self.transformed_grid = \
            FourierTransform.calculate_fourier_grid(self.original_latticevecs,
                                                    data_on_grid.shape)
        self.transformed_data = utilities.LinearInterpolationNOGrid(
            self.__calculate_fourier_transform(self.original_data),
            reciprocal_latticevecs)

    def __calculate_fourier_transform(self, original_data):
        return numpy.fft.fftn(original_data)

    @staticmethod
    def calculate_fourier_grid(latticevecs, data_shape):
        reciprocal_latticevecs = numpy.array([numpy.cross(latticevecs[1], latticevecs[2]),
                                               numpy.cross(latticevecs[2], latticevecs[0]),
                                               numpy.cross(latticevecs[0], latticevecs[1])
                                               ])
        reciprocal_latticevecs *= 2 * numpy.pi / numpy.linalg.det(latticevecs)

        gridsize = data_shape

        return reciprocal_latticevecs / data_shape, numpy.array([[[numpy.dot([i, j, k], reciprocal_latticevecs)
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
        for i, orb in real_space_orbitals.orbitals.iteritems():
            # XXX: if orb is given as analytic function: think of good gridsize
            self.fourier_transformations[i] = orb.fourier_transform()

    # XXX: write function to go from wave function in vector to array

    def fourier_transform(self, latticevecs, wave_function):
        """
        Coefficients of the wave function on a nonorthogonal grid (in a
        multidimensional array), one array per orbital.
        latticevecs: grid that spans the unit lattice
        """
        wave_function = numpy.array(wave_function)
        transformed_wave_function = numpy.fft.fftn(wave_function)
        transformed_lattice_vectors = FourierTransform.calculate_fourier_grid(latticevecs, wave_function.shape)

        transformed_coefficients = {}
        for orbnr, orb_transformation in self.fourier_transformations.iteritems():
            # XXX: if you transform several wave functions, you can save this
            transformed_orb_values_on_grid = numpy.array([[[
                        orb_transformation(point) for point in row]
                    for row in slice]
                for slice in transformed_lattice_vectors])

            transformed_coefficients[orbnr] = \
                transformed_wave_function * transformed_orb_values_on_grid

        return transformed_coefficients