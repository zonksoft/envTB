def list_to_molcas_block_format(mylist):
    vecchuncs=__chunks(mylist,4)
    strangeformat=[[__float_to_molcas_string(float(x)) for x in y] for y in vecchuncs]
    liste="\n".join(["".join(x) for x in strangeformat])
    return "".join(liste)

def __chunks(l, n):
    #http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
    return [l[i:i+n] for i in range(0, len(l), n)]

def __float_to_molcas_string(zahl):
    a='{:+018.12E}'.format(zahl)
    sign='0' if zahl>=0 else '-'
    exponent='E{:+03d}'.format(int(a[-3:])+1)
    zahl2='{:+015.12f}'.format(float(a[:15])/10)[2:]
    return sign+zahl2+exponent
    
def __generate_info(infostring, nrorbs):
    info = """*%s
%s%s%s
%s
%s"""%(infostring, 
       '0'.rjust(8), '1'.rjust(8), '0'.rjust(8), 
       str(nrorbs).rjust(8), 
       str(nrorbs).rjust(8))
    return info

def __generate_orbital_block(orbnr, coeffs):
    orbital_block = """* ORBITAL%s%s
%s"""%('1'.rjust(5), str(orbnr).rjust(5),
       list_to_molcas_block_format(coeffs))    
    return orbital_block

def __generate_one_electron_energy_block(one_electron_energies):
    one_electron_energy_block = """* ONE ELECTRON ENERGIES
%s"""%list_to_molcas_block_format(one_electron_energies)
    return one_electron_energy_block

def __generate_occupation_block(occupation_nrs):
    occupation_block = """* OCCUPATION NUMBERS
%s"""%list_to_molcas_block_format(occupation_nrs)
    return occupation_block

def __build_orbital_blocks(orbitals_coeffs):
    orbital_blocks = []
    for orbnr, coeffs in enumerate(orbitals_coeffs, 1):
        orbital_block = __generate_orbital_block(orbnr, coeffs)
        orbital_blocks.append(orbital_block)
        
    return orbital_blocks

def __concatenate_gssorb_file(**kwargs):
    """
    __concatenate_gssorb_file(info=info_section, orbitals=orbitals_section, ...)
    
    If an argument is None (i.e. orbitals=None), it will not be printed.
    """
    section_heads = [('info' , '#INFO'),
                     ('orbitals' , '#ORB'),
                     ('occupation' , '#OCC'),
                     ('one_electron' , '#ONE'),
                     ('index' , '#INDEX')]

    gssorb_file = '#INPORB 1.1\n'
    for key, val in section_heads:
        if key in kwargs and kwargs[key] is not None:
            gssorb_file += val + '\n' + kwargs[key] + '\n'
        
    return gssorb_file

def __concatenate_orbitals(orbital_blocks):
    return "\n".join(orbital_blocks)
    
def gssorb_writer(fname, infostring, orbitals_coefficients, occupations, one_electron_energies):
    nrorbs = len(orbitals_coefficients)
    orbital_blocks = __build_orbital_blocks(orbitals_coefficients)
    
    
    info_section = __generate_info(infostring, nrorbs)
    orbital_section = __concatenate_orbitals(orbital_blocks)  
    occupation_section = __generate_occupation_block(occupations)
    one_electron_energy_section = __generate_one_electron_energy_block(one_electron_energies)
    
    gssorb_file = open(fname,'w')    
    gssorb_file.write(__concatenate_gssorb_file(info=info_section, orbitals=orbital_section,
        occupation=occupation_section,one_electron=one_electron_energy_section))
    gssorb_file.close()
