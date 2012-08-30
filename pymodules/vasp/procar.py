class ProcarMap:
    """
    ProcarMap has functions returning the position of data in a VASP PROCAR file and saves the
    number of ions, bands and kpoints.
    
    i: kpoint index
    j: band index
    k: atom index
    All counters begin with 0.
    """
    __startoffset=3
    __morelinesperband=5
    __morelinesperkpoint=3
    __kpointoffset=2
    __bandoffset=3
    
    __nrions=0
    __nrbands=0
    __nrkpoints=0
    
    def __init__(self,procardataraw):
        """procardataraw is the VASP PROCAR file, read in the format [line.split() for line in file.readlines()]."""
        self.__nrkpoints,self.__nrbands,self.__nrions=self.__get_procar_system_data(procardataraw)
    
    def __get_procar_system_data(self,procardataraw):
        dataline = procardataraw[1]
        nrkpoints = int(dataline[3])
        nrbands = int(dataline[7])
        nrions = int(dataline[11])
        return nrkpoints,nrbands,nrions
      
    def kpunktstart(self,i):
        """Line where the data for kpoint i starts."""
        return self.__startoffset+i*((self.__morelinesperband+self.__nrions)*
               self.__nrbands+self.__morelinesperkpoint)
    
    def bandstart(self,i,j):
        """Line where the data for kpoint j in band i starts."""
        return self.kpunktstart(i)+self.__kpointoffset+j*(self.__nrions+
               self.__morelinesperband)
    
    def atomzeile(self,i,j,k):
        """Line with the data for ion k in band j in kpoint i."""
        return self.bandstart(i,j)+self.__bandoffset+k
    
    def nrions(self):
        """Number of ions"""
        return self.__nrions
    
    def nrbands(self):
        """Number of bands"""
        return self.__nrbands
    
    def nrkpoints(self):
        """Number of kpoints"""
        return self.__nrkpoints


class ProcarData:
    __procarMap=0
    
    __chargedata=0
    __energydata=0
    
    def __init__(self,filename):
        """filename is the name path to the VASP PROCAR file."""
        self.__LoadFromFile(filename)
    
    def __LoadFromFile(self,filename):
        procardataraw=self.__read_file_as_table(filename)
      
        self.__procarMap=ProcarMap(procardataraw)
      
        self.__chargedata=self.__get_procar_charge_data(procardataraw)
        self.__energydata=self.__get_procar_energy_data(procardataraw)    
    
    def __read_file_as_table(self,filename):
        filetoread = open(filename, 'r')
        data = [line.split() for line in filetoread.readlines()]
        filetoread.close()
        return data
    
    def __get_procar_charge_data(self,procardataraw):
        chargedata=[
                     [
                       [
                         [
                           float(x) for x in procardataraw[self.__procarMap.atomzeile(i,j,k)][1:9] 
                         ] for k in range(self.__procarMap.nrions())
                       ] for j in range(self.__procarMap.nrbands())
                     ] for i in range(self.__procarMap.nrkpoints())
                   ]
        return chargedata
    
    def __get_procar_energy_data(self,procardataraw):
        energydata=[
                     [
                       float(procardataraw[self.__procarMap.bandstart(i, j)][4]) for j in range(self.__procarMap.nrbands())
                     ] for i in range(self.__procarMap.nrkpoints())
                   ]
        return energydata
    
    def info(self):
        """Number of bands, kpoints and ions."""
        return self.__procarMap.nrbands(),self.__procarMap.nrkpoints(),self.__procarMap.nrions()   
    
    def chargedata(self):
        """PROCAR projection charge data as a four-dimensional array with 
         the indices data[kpoint][band][ion][orbital]"""
        return self.__chargedata
    
    def energydata(self):
        """PROCAR energy data as a two-dimensional array with the 
           indices data[kpoint][band]"""
        return self.__energydata  
    
