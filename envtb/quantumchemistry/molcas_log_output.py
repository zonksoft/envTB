import re
import numpy

def _numpy_array_from_triangle(matrix_triangle):
    """
    matrix_triangle: lower triangle
    """
    nrorbs = len(matrix_triangle[-1])
    matrix = numpy.zeros((nrorbs, nrorbs))        
    for i, line in enumerate(matrix_triangle):
        for j, elem in enumerate(line):
            matrix[i,j] = elem
            matrix[j,i] = elem
    return matrix
        
        
class MolcasFockMatrix:
    def __init__(self, fock_matrix):
        self.fock_matrix = numpy.array(fock_matrix, copy=False)
        
    @classmethod
    def from_molcas_output(cls, molcas_file, nrorbs):
        """
        molcas_file: Log file contents as list of strings
        nrorbs: number of basis orbitals
        """
        
        return cls(MolcasFockMatrix.__get_fock_matrix(molcas_file, nrorbs))
    
    @staticmethod
    def __get_fock_matrix(molcas_file, nrorbs):
        """
        Returns the fock matrix (the first one that occurs in the file).
        """
        search_pattern=re.compile('print Fock-Matrix in AO-basis')
        
        for nr, line in enumerate(molcas_file):
            res=search_pattern.search(line)
            if res:
                matrix_lines = molcas_file[nr+1:nr+1+nrorbs]
                matrix_triangle = [[float(x) for x in line.split()] for line in matrix_lines]
                
                matrix = _numpy_array_from_triangle(matrix_triangle)
                return matrix    
                
    def apply_operator(self, left, right):
        """
        Returns the matrix product left*matrix*right. left and right can be vectors
        or matrices. If you want to apply the matrix to many vectors, put them into
        "left" rowwise and into "right" columnwise.
        """
        
        return numpy.dot(left, numpy.dot(self.fock_matrix, right))

    def N(self):
        return self.fock_matrix.shape[0]
                
                
class MolcasOutput:
    def __init__(self, path):
        self.molcas_file = open(path).readlines()
        self.number_of_basis_functions, self.occupied_orbitals_alpha, self.occupied_orbitals_beta, \
            self.orbital_file_label = self.__get_initial_properties(self.molcas_file)
        self.total_scf_energy, self.one_electron_energy, self.two_electron_energy = \
            self.__get_scf_results(self.molcas_file)
        
        self.overlap_matrix = self.__get_overlap_matrix(self.molcas_file, self.number_of_basis_functions)
        self.fock_matrix = MolcasFockMatrix.from_molcas_output(
            self.molcas_file, self.number_of_basis_functions)        
        
    def __regex_patterns_through_list(self, patterns, data, datatype=float):
        """
        Takes a list of regex patterns which contain one group each,
        goes through a list of strings and returns the results as a list.
        
        patterns: list of regex pattern strings which contain one
                  group. If found, the group will be converted to float.
        data: list of strings to search through.
        datatype: type (float, int or str) to convert the result to
                  
        Return:
        results: list of resulting floats corresponding to the patterns
                 If not found, the list element will be None.
        """
        re_compiled=[re.compile(x) for x in patterns]
        results = [None for x in patterns]
        
        for nr, line in enumerate(data):
            for i, search_pattern in enumerate(re_compiled):
                res=search_pattern.search(line)
                if res:
                    results[i] = datatype(res.group(1).strip())
        return results        
        
    def __get_scf_results(self, molcas_file):
        patterns = ('Total SCF energy(.*)', 'One-electron energy(.*)',
                    'Two-electron energy(.*)')
        
        return self.__regex_patterns_through_list(patterns, molcas_file, float)
                
    def __get_initial_properties(self, molcas_file):
        patterns = ('Total number of orbitals(.*)', 'Occupied orbitals alpha(.*)',
                    'Occupied orbitals beta(.*)')
        
        str_patterns = ('Orbital file label:(.*)',)
        
        return self.__regex_patterns_through_list(patterns, molcas_file, int) + \
            self.__regex_patterns_through_list(str_patterns, molcas_file, str)      
        
    def __get_overlap_matrix(self, molcas_file, nrorbs):
        """
        Returns the overlap matrix (the first one that occurs in the file).
        """
        search_pattern=re.compile('print overlap matrix produced by seward')
        
        for nr, line in enumerate(molcas_file):
            res=search_pattern.search(line)
            if res:
                matrix_lines = molcas_file[nr+1:nr+1+nrorbs]
                matrix_triangle = [[float(x) for x in line.split()] for line in matrix_lines]
                
                matrix = _numpy_array_from_triangle(matrix_triangle)
                return matrix
            
        
