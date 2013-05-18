class OutcarData:
    def __init__(self, fname):
        self.fname = fname
        self.efermi = self.__read_fermi_energy(fname)
        
    def __read_fermi_energy(self,outcarfilename):
        f = open(outcarfilename, 'r')
        lines = f.readlines()
        
        for nr,line in enumerate(lines):
            ret = line.find("E-fermi :")
            if ret >=0:
                break   
                
        fermi_energy=float(lines[nr].split()[2])
        
        return fermi_energy
