from envtb.quantumchemistry.molden import FileSectionUtilities
import numpy

# XXX: merge with MolecularOrbitalSet in molden.py

class InpOrbReader:
    @staticmethod
    def split_len(seq, length):
        return [seq[i:i+length] for i in range(0, len(seq), length)]

    def __load_inporb_file(self, fname):
        inporb_file = [line.strip() for line in open(fname).readlines()]
        return inporb_file
    
    def __parse_molecular_coefficients(self, mo_section):
        mo_section_heads = FileSectionUtilities.get_section_heads_and_positions(mo_section, '\* ORBITAL')
        mos = FileSectionUtilities.split_by_sections(mo_section, mo_section_heads)
        
        mos_coeffs = []
        for mo in mos:
            mo_coeffs = []
            for line in mo[1:]:
                mo_coeffs.extend([float(x) for x in InpOrbReader.split_len(line, 18)])
            mos_coeffs.append(mo_coeffs)
        return numpy.array(mos_coeffs)        
    
    def __get_inporb_section_heads(self, inporb_file):
        return FileSectionUtilities.get_section_heads_and_positions(inporb_file, '#')
    
    def __get_mo_section(self, inporb_file, inporb_section_heads):
        return FileSectionUtilities.get_section(inporb_file, inporb_section_heads, '#ORB')
        
    def __get_uhf_mo_section(self, inporb_file, inporb_section_heads):
        return FileSectionUtilities.get_section(inporb_file, inporb_section_heads, '#UORB')        
    
    def __parse_inporb_file(self, inporb_file):
        inporb_section_heads = self.__get_inporb_section_heads(inporb_file)
        mo_section = self.__get_mo_section(inporb_file, inporb_section_heads)
        uhf_mo_section = self.__get_uhf_mo_section(inporb_file, inporb_section_heads)        
        
        molecular_orbitals = self.__parse_molecular_coefficients(mo_section)
        uhf_molecular_orbitals = self.__parse_molecular_coefficients(uhf_mo_section)        
        
        return molecular_orbitals, uhf_molecular_orbitals
        
    def __init__(self, fname):
        inporb_file = self.__load_inporb_file(fname)
        self.__molecular_orbitals, self.__uhf_molecular_orbitals = self.__parse_inporb_file(inporb_file)

    def molecular_orbitals(self, spin='alpha', order=None):
        if spin == 'alpha':
            orbs = self.__molecular_orbitals
        elif spin == 'beta':
            if self.__uhf_molecular_orbitals is None:
                ValueError('beta spin is not in file')
            else:
                orbs = self.__uhf_molecular_orbitals
        else:
            ValueError('%s is not a valid spin name' % spin)
            
        if order is None:
            return orbs
        else:
            return [[orbital[i] for i in order] for orbital in orbs]
