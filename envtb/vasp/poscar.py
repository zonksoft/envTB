import general
import math
import numpy as np

class PoscarData:
    """
    The class supplies an interface to the data contained
    in a VASP POSCAR file, which is:
    - lattice vectors: Contained in the object lattice_vectors (see 
      class LatticeVectors), use e.g. my_poscar_data.lattice_vectors.latticevecs() 
    - atom positions (output in cartesian and direct coordinates)
    """
    
    lattice_vectors=0
    
    def __init__(self,filename):
        """
        filename: Path to POSCAR file.
        """
        self.__read_file(filename)
        
    def __read_file(self,filename):
        dataraw=general.read_file_as_table(filename)
        latticeconstant=float(dataraw[1][0])
        data=np.array(dataraw[2:5]).astype(np.float)
        
        self.lattice_vectors=LatticeVectors(data,latticeconstant)
    
    """
    TODO: finish class
    """
    
class LatticeVectors:
    """
    The class contains a set of crystal lattice vectors, can calculate the
    reciprocal lattice vectors and can convert direct reciprocal coordinates to
    cartesian reciprocal coordinates.
    """
    
    __latticevecs=[]
    __reciprocal_latticevecs=[]    
    
    def __init__(self,latticevecs,latticeconstant=1):
        """
        Initializes the object with latticevecs (list of lattice vectors).
        latticeconstant is a scaling factor (optional, default 1).
        """
        
        self.__latticevecs = np.array(latticevecs)*latticeconstant
    
    def latticevecs(self):
        """
        Returns the lattice vectors.
        """
        return self.__latticevecs
    
    def reciprocal_latticevecs(self):
        """
        Returns the reciprocal lattice vectors.
        """
        latticevecs=self.__latticevecs
        if self.__reciprocal_latticevecs==[]:
            reciprocal_latticevecs=np.array([np.cross(latticevecs[1], latticevecs[2]),
                                               np.cross(latticevecs[2], latticevecs[0]),
                                               np.cross(latticevecs[0], latticevecs[1])
                                               ])
            reciprocal_latticevecs*=2*math.pi/np.linalg.det(self.__latticevecs)
            self.__reciprocal_latticevecs=reciprocal_latticevecs
            
        return self.__reciprocal_latticevecs   
    
    def direct_to_cartesian_reciprocal(self,k):
        """
        Converts direct reciprocal coordinates to cartesian reciprocal coordinates.
        
        k: kpoint or list of kpoints in direct coordinates
        """
        if isinstance(k[0],list):
            return [self.direct_to_cartesian_reciprocal(thisk) for thisk in k]
        reclattice_transposed=self.reciprocal_latticevecs().transpose()
        return np.dot(reclattice_transposed,k)     
    