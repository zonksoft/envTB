import general
import numpy

class EigenvalData:
    """
    Reads the VASP EIGENVAL file and makes the
    contents easily accessible.
    """
    
    __nrelectrons=0
    __nrkpoints=0
    __nrbands=0
    __kpoints=numpy.array([])
    __energies=numpy.array([])
    
    def __init__(self,filename):
        """
        filename: Path to the VASP EIGENVAL file.
        """
        self.__nrelectrons,self.__nrkpoints,self.__nrbands,self.__kpoints,self.__energies=self.__read_file(filename)
        
    def __read_file(self,filename):
        data = general.read_file_as_table(filename)
        nrelectrons,nrkpoints,nrbands=[int(x) for x in data[5]]

        kpoints=[]
        energies=[]
        #Not so nice to read
        for kpointfirstline in [7+kpoint*(nrbands+2) for kpoint in range(nrkpoints)]:
            kpoints.append([float(x) for x in data[kpointfirstline][:3]])
            energies.append([float(x[1]) for x in data[kpointfirstline+1:kpointfirstline+nrbands+1]])
            
        return nrelectrons,nrkpoints,nrbands,numpy.array(kpoints),numpy.array(energies)
    
    def kpoints(self):
        """
        Returns a list of the kpoints.
        """
        return numpy.array(self.__kpoints)
    
    def energies(self):
        """
        Returns a list of the eigenvalues at each kpoint. Use the kpoint as the
        first and the band number as the second list index.
        """
        return numpy.array(self.__energies)
    
    def nrelectrons(self):
        """
        Returns the number of electrons in the system.
        """
        return self.__nrelectrons
    
    def nrkpoints(self):
        """
        Returns the number of kpoints.
        """        
        return self.__nrkpoints
    
    def nrbands(self):
        """
        Returns the number of bands.
        """            
        return self.__nrbands