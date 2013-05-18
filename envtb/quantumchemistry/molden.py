import re
import envtb.general
import numpy

#List of Numpy arrays:
#
#Molecular orbital coefficients


#Idee: Spezialfaelle (zB PAH-Molden-file) ableiten und mit extra fkt ausstatten (zB select_pi_system())

class FileSectionUtilities:
    @staticmethod
    def get_section_heads_and_positions(lines, section_re):
        re_filter = re.compile(section_re)
        sections = [[i, line] for i, line in enumerate(lines) 
                              if re_filter.match(line)]
        return sections
    
    @staticmethod
    def get_section(lines, sections, section_name):            
        for i in range(len(sections)):
            line_nr = sections[i][0]
            line = sections[i][1]
            if re.match('^'+re.escape(section_name),line):
                start = line_nr
                try:
                    end = sections[i+1][0]-1
                except IndexError:
                    end = len(lines)-1
                return lines[start:end+1]
        raise ValueError('%s section expected, not found!'%section_name)
    
    @staticmethod    
    def split_by_sections(lines, section_heads_and_positions):
        blocks = []
        for i in range(len(section_heads_and_positions)):
            line_nr = section_heads_and_positions[i][0]
            try:
                next_line_nr = section_heads_and_positions[i+1][0]-1
            except IndexError:
                next_line_nr = len(lines)-1
            blocks.append(lines[line_nr:next_line_nr+1])
        return blocks
        
# XXX: info function for each hierarchy step (atom, gaussian, gto set...)
# XXX: after reading, create one orbital for each angular momentum instead of one total,
# rewrite concatenate_orbitals and OrbitalList. Make selection for L^2, m, element
# XXX: merge AtomicOrbitalsGTO/GaussianOrbitalAtomSet and Atom. Create AtomSet.
# XXX: better names: p = L^2, px = m (at least similar)
# XXX: make PAH a special case of MolecularOrbitalSet (or similar), add select_pi_system() function
# for orbital coefficients and implement Factory to select between normal and PAH system from the MoldenFile
# constructor

class Atom:
    def __init__(self, name, nr, elem, x, y, z):
        self.name, self.nr, self.elem = name, nr, elem
        self.coord = [x,y,z]
        
    @classmethod
    def from_molden_line(cls, line):
        split = [envtb.general.string_to_number(item) for item in line.split()]
        return cls(*split)
        
    def __repr__(self):
        return '<atom %s %i %s>'%(self.name,self.nr,str(self.coord))
    
class MolecularOrbital:
    def __init__(self, symmetry, energy, spin, occupation, coefficients):
        self.symmetry, self.energy, self.spin, self.occupation, self.coefficients= \
            symmetry, energy, spin, occupation, numpy.array(coefficients,copy=False)
            
    def __repr__(self):
        return '<molecular orbital %s en %e occ %e>'%(self.symmetry,self.energy, 
                                                      self.occupation)
            
    @classmethod
    def from_molden_file(cls, lines):
        symmetry = re.match('Sym(?:.*)=(.*)',lines[0]).groups()[0].strip()
        energy = float(re.match('Ene(?:.*)=(.*)',lines[1]).groups()[0].strip())
        spin = re.match('Spin(?:.*)=(.*)',lines[2]).groups()[0].strip()
        occupation = float(re.match('Occup(?:.*)=(.*)',lines[3]).groups()[0].strip())
        
        coefficients = [float(line.split()[1]) for line in lines[4:]]
        
        return cls(symmetry, energy, spin, occupation, coefficients)    
    
class MolecularOrbitalSet:
    def __init__(self, molecular_orbitals):
        self.molecular_orbitals = molecular_orbitals
        
    @classmethod
    def from_molden_file(cls, mo_section):
        #inklusive [MO] zeile!
        mo_section_heads = \
            FileSectionUtilities.get_section_heads_and_positions(mo_section, 'Sym=')
        mo_orbital_blocks = \
            FileSectionUtilities.split_by_sections(mo_section, mo_section_heads)
        molecular_orbitals = \
            [MolecularOrbital.from_molden_file(block) for block in mo_orbital_blocks]
        
        return cls(molecular_orbitals)
    
    def plot_occupations(self, ax):
        occupations = [orb.occupation for orb in self.molecular_orbitals]
        dots = ax.plot(occupations,'.')
        line = ax.plot(occupations,'-')
        
        ax.set_xlabel('Orbital number')
        ax.set_ylabel('Occupation')
        
        return dots, line
    
class MoldenFile:
    def __load_molden_file(self, fname):
        molden_file = [line.strip() for line in open(fname).readlines()]
        return molden_file
    
    def __get_molden_file_sections(self, molden_file):
        return FileSectionUtilities.get_section_heads_and_positions(molden_file,'^\[')
    
    def __get_atoms(self, molden_file, sections):
        atoms_section = \
            FileSectionUtilities.get_section(molden_file, sections, '[Atoms]')
        units = atoms_section[0].split()[1]
        
        atoms = [Atom.from_molden_line(line) for line in atoms_section[1:]]
    
        return atoms, units
    
    def __get_mo_section(self, molden_file, file_sections):
        return FileSectionUtilities.get_section(molden_file, file_sections, '[MO]')
    
    def __get_gto_section(self, molden_file, file_sections):
        return FileSectionUtilities.get_section(molden_file, file_sections, '[GTO]')    
    
    def __parse_molden_file(self,molden_file):
        global gto_section
        file_sections = self.__get_molden_file_sections(molden_file)
        atoms, length_unit = self.__get_atoms(molden_file, file_sections)
        mo_section = self.__get_mo_section(molden_file, file_sections)
        molecular_orbitals = MolecularOrbitalSet.from_molden_file(mo_section)
        
        gto_section = self.__get_gto_section(molden_file, file_sections)
        atomic_orbitals = AtomicOrbitalsGTO(gto_section)
        
        return atoms, length_unit, atomic_orbitals, molecular_orbitals 

    def __init__(self, fname):
        molden_file = self.__load_molden_file(fname)
        
        self.atoms, self.length_unit, self.atomic_orbitals, self.molecular_orbitals = \
            self.__parse_molden_file(molden_file)   
            
            
class GaussianOrbitalAtomSet:
    def __init__(self, atom_number, gaussian_orbitals):
        self.gaussian_orbitals = gaussian_orbitals
        self.atom_number = atom_number
        
class GaussianOrbital:
    def __init__(self, orb_type, coefficients):
        self.orb_type, self.coefficients, self.nr_gaussians = orb_type, coefficients, len(coefficients)
        self.angular_momentum_list = self.__angular_momentum_list(self.orb_type)
        
    def __angular_momentum_list(self, orb_type):
        table = {'s' : ['s'],
                 'p' : ['px','py','pz'],
                 'd' : ['d1','d2','d3','d4','d5'],
                 'f' : ['f1','f2','f3','f4','f5','f6','f7']}
        
        return table[orb_type]

class AtomicOrbitalsGTO:
    @staticmethod
    def __scient_d_str_to_float(x):
        return float(x.replace('D','E'))
    
    def __parse_gto_blocks(self, lines):
        gto_sections_heads = FileSectionUtilities.get_section_heads_and_positions(lines, '([0-9]*)  ([0-9]*)')
        gto_sections = FileSectionUtilities.split_by_sections(lines, gto_sections_heads)
        
        gto_blocks = []
        for gto_section in gto_sections:
            orb_sections_heads = \
                FileSectionUtilities.get_section_heads_and_positions(gto_section, '[spdfgh]')
            orb_sections = \
                FileSectionUtilities.split_by_sections(gto_section, orb_sections_heads)
            gto_blocks.append(orb_sections)
            
        return gto_blocks, gto_sections_heads
    
    def __parse_orbital_blocks(self, atom_number, gto_block):
        gaussian_orbitals = []
        for orb_block in gto_block:
            orb_type, nr_gaussians, _ = [envtb.general.string_to_number(x) for x in orb_block[0].split()]
            gaussian_coeffs = [[AtomicOrbitalsGTO.__scient_d_str_to_float(x) for x in coeff_line.split()] for coeff_line in orb_block[1:] if coeff_line != '']
            gaussian_orbitals.append(GaussianOrbital(orb_type, gaussian_coeffs))
        return GaussianOrbitalAtomSet(atom_number, gaussian_orbitals)        
    
    def __init__(self, lines):
        gto_blocks, gto_sections_heads = self.__parse_gto_blocks(lines)
        atom_numbers = [int(x[1].split()[0]) for x in gto_sections_heads]
        self.atom_sets = [self.__parse_orbital_blocks(atom_number, gto_block) for atom_number, gto_block in zip(atom_numbers, gto_blocks)]
        
    def concatenate_orbitals(self):
        orbitals = []
        for atom in self.atom_sets:
            for gaussian_nr, gaussian_orbital in enumerate(atom.gaussian_orbitals):
                for angular_momentum_nr, angular_momentum_name in enumerate(gaussian_orbital.angular_momentum_list):
                    orbitals.append([atom.atom_number, gaussian_nr, gaussian_orbital.orb_type, angular_momentum_name])
        return OrbitalList(orbitals)
    
class OrbitalList:
    def __init__(self, orbitals):
        self.orbitals = orbitals
        
    def select_angular_momentum(self, angular_momentum, numpy_filter_mode=False):
        #angular momentum acc. to __angular_momentum_list table (s, px, py, pz etc.)
        #use numpy_filter_mode like: orbital_coeffs[selection]
        if numpy_filter_mode:
            return numpy.array([orbital[3] == angular_momentum for orbital in self.orbitals])
        else:
            return [i for i, orbital in enumerate(self.orbitals) if orbital[3] == angular_momentum]
                                 
