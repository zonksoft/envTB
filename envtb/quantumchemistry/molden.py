import re
import envtb.general
import numpy
import math
import envtb.quantumchemistry.gssorb_writer
from envtb.quantumchemistry import disentanglement

#List of Numpy arrays:
#
#Molecular orbital coefficients

#Enumeration:
#Everything starts at zero, except for the atom numbers read from the molden file, accessible
#through Atom.nr.


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
# rewrite concatenate_orbitals and OrbitalList (especially unclean!!) and SimpleMolecularOrbitalPlotter. Make selection for L^2, m, element
# the plot function should supply this info internally.
# XXX: merge AtomicOrbitalsGTO/GaussianOrbitalAtomSet and Atom. Create AtomSet.
# XXX: better names: p = L^2, px = m (at least similar)
# XXX: make PAH a special case of MolecularOrbitalSet (or similar), add select_pi_system() function
# for orbital coefficients and implement Factory to select between normal and PAH system from the MoldenFile
# constructor
# XXX: MolecularOrbital.normalize() has an unclean copy operation
# XXX: use orbital type in SimpleMolecularOrbitalPlotter
# XXX: molecular orb shall remember his number and orb set
# XXX: instead of saving orbital nr in MolecularOrbital: save MolecularOrbitalSet (might not work if a MolecularOrbital is used in more than one Set... a list is necessary! phew...) and
# make function nr() that requests position in list
# XXX: link orbital geometry and atomic orbitals with MolecularOrbital
# XXX: encapsulate occupation and energy from MolecularOrbital, because they might change if the orbital is used differently
# XXX: link norm calculation MolecularOrbitalSet.norm(mo) to MolecularOrbital.norm(). Independent of the set an orbital belongs to, the norm is always the same!

class Atom:
    def __init__(self, name, nr, elem, x, y, z):
        self.name, self.nr, self.elem = name, nr, elem
        self.coordinates = [x,y,z]
        
    @classmethod
    def from_molden_line(cls, line):
        split = [envtb.general.string_to_number(item) for item in line.split()]
        return cls(*split)
        
    def __repr__(self):
        return '<atom %s %i %s>'%(self.name,self.nr,str(self.coord))
    
class MolecularOrbital:
    def __init__(self, nr, symmetry, energy, spin, occupation, coefficients):
        self.nr, self.symmetry, self.energy, self.spin, self.occupation, self.coefficients= \
            nr, symmetry, energy, spin, occupation, numpy.array(coefficients,copy=False)
            
    def __repr__(self):
        return '<molecular orbital %s en %s occ %s nr %s>'%(str(self.symmetry),str(self.energy), 
                                                      str(self.occupation),str(self.nr))
            
    @classmethod
    def from_molden_file(cls, nr, lines):
        symmetry = re.match('Sym(?:.*)=(.*)',lines[0]).groups()[0].strip()
        energy = float(re.match('Ene(?:.*)=(.*)',lines[1]).groups()[0].strip())
        spin = re.match('Spin(?:.*)=(.*)',lines[2]).groups()[0].strip()
        occupation = float(re.match('Occup(?:.*)=(.*)',lines[3]).groups()[0].strip())
        
        coefficients = [float(line.split()[1]) for line in lines[4:]]
        
        return cls(nr, symmetry, energy, spin, occupation, coefficients)
        
    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise NotImplementedError        
        return self.__class__(None, None, None, None, None,self.coefficients+other.coefficients)
        
    def __sub__(self, other):
        if not isinstance(other, self.__class__):
            raise NotImplementedError    
        return self.__class__(None, None, None, None, None,self.coefficients-other.coefficients)  
        
    def __mul__(self, other):
        if not (isinstance(other, int) or isinstance(other, float)):
            raise NotImplementedError
        return self.__class__(self.nr, self.symmetry, self.energy, self.spin, self.occupation,self.coefficients*other)
        
    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __div__(self, other):
        if not (isinstance(other, int) or isinstance(other, float)):
            raise NotImplementedError    
        return self.__mul__(1./other)
                    
        
class SimpleMolecularOrbitalPlotter:
    def __init__(self, molecule_geometry, atomic_orbitals, scale=500, colors=('blue','red')):
        self.molecule_geometry = molecule_geometry
        self.atomic_orbitals = atomic_orbitals
        self.scale = scale
        self.colors = colors
        
    def plot(self, ax, orbital):
        atom_coordinates = self.molecule_geometry.coordinates()
        
        x = []
        y = []
        s = []
        c = []
        
        for orb, coeff in zip(self.atomic_orbitals.concatenate_orbitals().orbital_properties, orbital.coefficients):
            atom_number = orb[0]
            orb_type = orb[3]
            atom_coordinate = atom_coordinates[atom_number - 1]
            x.append(atom_coordinate[0])
            y.append(atom_coordinate[1])  
            s.append(abs(coeff)*self.scale)          
            if coeff > 0:
                c.append('red')
            else:
                c.append('blue')
        
        ax.set_xlabel('x [%s]'%self.molecule_geometry.units)
        ax.set_ylabel('y [%s]'%self.molecule_geometry.units)
        return ax.scatter(x, y, s, c, edgecolors='none')
            
             

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
            [MolecularOrbital.from_molden_file(nr, block) for nr, block in enumerate(mo_orbital_blocks)]
        
        return cls(molecular_orbitals)
        
    def occupations(self):
        return [orb.occupation for orb in self.molecular_orbitals]
    
    def plot_occupations(self, ax):
        occupations = self.occupations()
        line = ax.plot(occupations,marker='.', linestyle='-',markersize=2)
        
        ax.set_xlabel('Orbital number')
        ax.set_ylabel('Occupation')
        
        return line
        
    def __getitem__(self, key):
        return self.molecular_orbitals[key]
        
    def overlap(self):
        """
        Overlap matrix is S=(V^T)^-1 V^-1
        V is the transformation matrix with the molecular orbitals as columns.
        
        Note that molecular_orbital_matrix() gives V^T.
        """
        transformation_matrix = self.molecular_orbital_matrix()
        return numpy.dot(numpy.linalg.inv(transformation_matrix),numpy.linalg.inv(numpy.transpose(transformation_matrix)))
        
    def inner_product(self, orb1, orb2):
        """
        orb1 and orb2 can be orbital object or orbital number!
        """
        if isinstance(orb1, int):
           orb1 = self.molecular_orbitals[orb1]
        if isinstance(orb2, int):
           orb2 = self.molecular_orbitals[orb2] 
                     
        coeff1 = orb1.coefficients
        coeff2 = orb2.coefficients
        return numpy.dot(numpy.dot(coeff1, self.overlap()), coeff2)
    
    def molecular_orbital_matrix(self):
        #molecular orbitals in rows
        return numpy.vstack([orb.coefficients for orb in self.molecular_orbitals])
        
    def norm(self, orb):
        return math.sqrt(self.inner_product(orb, orb))
        
    def save_to_molcas_uhforb_file(self, other_mo_set, fname, infostring, add_dummy_energies=False):
        """
        An UhfOrb file contains two molecular orbital sets, so another set has to be supplied
        through other_mo_set. The current set will be the first set written to the file.
        
        fname: *.UhfOrb
        add_dummy_energies: Creates dummy energies in steps of 0.1 eV instead of
        the energies saved in the instance (which are probably zero if you are
        using this feature)        
        """
        
        orbitals_coefficients_1 = [orb.coefficients for orb in self.molecular_orbitals]
        occupations_1 = [orb.occupation for orb in self.molecular_orbitals]
        
        orbitals_coefficients_2 = [orb.coefficients for orb in other_mo_set.molecular_orbitals]
        occupations_2 = [orb.occupation for orb in other_mo_set.molecular_orbitals]
                
        if not add_dummy_energies:
            one_electron_energies = [orb.energy for orb in self.molecular_orbitals]
        else:
            nr_orbitals = len(self.molecular_orbitals)
            one_electron_energies = numpy.arange(-nr_orbitals/2., nr_orbitals/2.)*0.1  
        
        envtb.quantumchemistry.gssorb_writer.uhforb_writer(
            fname, infostring,
            orbitals_coefficients_1, occupations_1, one_electron_energies,
            orbitals_coefficients_2, occupations_2, one_electron_energies)            
                  
    
    def save_to_molcas_gssorb_file(self, fname, infostring, add_dummy_energies=False):
        """
        fname: *.GssOrb

        add_dummy_energies: Creates dummy energies in steps of 0.1 eV instead of
        the energies saved in the instance (which are probably zero if you are
        using this feature)
        """
        
        orbitals_coefficients = [orb.coefficients for orb in self.molecular_orbitals]
        occupations = [orb.occupation for orb in self.molecular_orbitals]
        if not add_dummy_energies:
            one_electron_energies = [orb.energy for orb in self.molecular_orbitals]
        else:
            nr_orbitals = len(self.molecular_orbitals)
            one_electron_energies = numpy.arange(-nr_orbitals/2., nr_orbitals/2.)*0.1
        envtb.quantumchemistry.gssorb_writer.gssorb_writer(
            fname, infostring, orbitals_coefficients, occupations, 
            one_electron_energies)
            
    def disentangle(self, orbitals_to_disentangle, create_two_sets=False):
        """
        Returns one or two MolecularOrbitalSets with disentangled orbitals.
        Only the properties of the disentangled orbitals are changed.
        
        create_two_sets: Two different modes for the new orbital set.
            create_two_sets=False: Create one orbital set. The disentangled orbitals
            will directly substitute the original orbitals.
            create_two_sets=True: Create two orbital sets, corresponding to up and
            down spin in an UHF calculation. The positive linear combinations go in the
            first spin, the negative in the second spin. 
            (Note: The upper half of the original orbitals remain, but they are unoccupied.)
        """
        
        disentangled_orbs, _ = disentanglement.disentangle_mo_set(orbitals_to_disentangle)
                    
        if create_two_sets is True:
            new_mo_list_1 = list(self.molecular_orbitals) # Copy the original list
            new_mo_list_2 = list(self.molecular_orbitals) 
            
            for orig, dis in zip(orbitals_to_disentangle[:len(disentangled_orbs)], disentangled_orbs): # Associate the first half of the original orbital with the elements of disentangled_orbs
                new_mo_list_1[new_mo_list_1.index(orig)] = dis[0] #positive linear combination
                new_mo_list_2[new_mo_list_2.index(orig)] = dis[1] #negative linear combination
                
            return MolecularOrbitalSet(new_mo_list_1), MolecularOrbitalSet(new_mo_list_2)        
            
        else:
            disentangled_orbs_flat = [orb for orbpair in disentangled_orbs for orb in orbpair] # implicitly sorts the disentangled orbitals. here: a0b0+, a0b0-, a1b1+, ...
            new_mo_list = list(self.molecular_orbitals) # Copy the original list
            
            for orig, dis in zip(orbitals_to_disentangle, disentangled_orbs_flat): # Associate each original orbital with a disentangled orbital and substitute
                new_mo_list[new_mo_list.index(orig)] = dis
                
            return MolecularOrbitalSet(new_mo_list)
        
        
    def print_orbital_list(self):
        print self.molecular_orbitals
        

class MoleculeGeometry:
    def __init__(self, atom_list, units):
        self.atom_list = atom_list
        self.units = units
        
    def __getitem__(self, key):
        return self.atom_list[key]
        
    def coordinates(self):
        return [atom.coordinates for atom in self.atom_list]
            
        
class MoldenFile:
    def __load_molden_file(self, fname):
        molden_file = [line.strip() for line in open(fname).readlines()]
        return molden_file
    
    def __get_molden_file_sections(self, molden_file):
        return FileSectionUtilities.get_section_heads_and_positions(molden_file,'^\[')
    
    def __get_atoms(self, molden_file, sections):
        try:
            atoms_section = \
                FileSectionUtilities.get_section(molden_file, sections, '[Atoms]')
        except ValueError:
            atoms_section = \
                FileSectionUtilities.get_section(molden_file, sections, '[ATOMS]')            
        units = atoms_section[0].split()[1]
        
        molecule_geometry = MoleculeGeometry([Atom.from_molden_line(line) for line in atoms_section[1:]], units)
    
        return molecule_geometry
    
    def __get_mo_section(self, molden_file, file_sections):
        return FileSectionUtilities.get_section(molden_file, file_sections, '[MO]')
    
    def __get_gto_section(self, molden_file, file_sections):
        return FileSectionUtilities.get_section(molden_file, file_sections, '[GTO]')    
    
    def __parse_molden_file(self,molden_file):
        global gto_section
        file_sections = self.__get_molden_file_sections(molden_file)
        molecule_geometry = self.__get_atoms(molden_file, file_sections)
        mo_section = self.__get_mo_section(molden_file, file_sections)
        molecular_orbitals = MolecularOrbitalSet.from_molden_file(mo_section)
        
        gto_section = self.__get_gto_section(molden_file, file_sections)
        atomic_orbitals = AtomicOrbitalsGTO(gto_section)
        
        return molecule_geometry, atomic_orbitals, molecular_orbitals 

    def __init__(self, fname):
        molden_file = self.__load_molden_file(fname)
        
        self.molecule_geometry, self.atomic_orbitals, self.molecular_orbitals = \
            self.__parse_molden_file(molden_file)
            
            
class GaussianOrbitalAtomSet:
    def __init__(self, atom_number, gaussian_orbitals):
        self.gaussian_orbitals = gaussian_orbitals
        self.atom_number = atom_number
        
    def __getitem__(self, key):
        return self.gaussian_orbitals[key]
        
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
        
    def __getitem__(self, key):
        return self.atom_sets[key]
        
    def concatenate_orbitals(self):
        orbital_properties = []
        for atom in self.atom_sets:
            for gaussian_nr, gaussian_orbital in enumerate(atom.gaussian_orbitals):
                for angular_momentum_nr, angular_momentum_name in enumerate(gaussian_orbital.angular_momentum_list):
                    orbital_properties.append([atom.atom_number, gaussian_nr, gaussian_orbital.orb_type, angular_momentum_name])
        return OrbitalList(orbital_properties)
    
class OrbitalList:
    def __init__(self, orbital_properties):
        self.orbital_properties = orbital_properties
        
    def select_angular_momentum(self, angular_momentum, numpy_filter_mode=False):
        #angular momentum acc. to __angular_momentum_list table (s, px, py, pz etc.)
        #use numpy_filter_mode like: orbital_coeffs[selection]
        if numpy_filter_mode:
            return numpy.array([orbital[3] == angular_momentum for orbital in self.orbital_properties])
        else:
            return [i for i, orbital in enumerate(self.orbital_properties) if orbital[3] == angular_momentum]
                                 
