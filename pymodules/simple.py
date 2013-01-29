from vasp import eigenval
from vasp import procar
from wannier90 import w90hamiltonian
import numpy

"""
This file contains functions for often used procedures.
They can also be considered as use cases.

It's also recommended to use them as starting point for your own
functions: type ?? after the function name and press enter,
then you can copy, directly modify and execute the code. 
"""

def plot_vasp_bandstructure(eigenval_filename,plot_filename,output='save'):
    """
    Plot the bandstructure contained in a VASP EIGENVAL file.
    The output format is determined by the file ending of
    plot_filename.
    
    output: determines if the plot is written to a file ('save' - default value) or
    displayed ('show')
    """
    
    eigenvaldata=eigenval.EigenvalData(eigenval_filename)
    kpoints=eigenvaldata.kpoints()
    energies=eigenvaldata.energies()
    plotter=w90hamiltonian.BandstructurePlot()
    plotter.plot(kpoints,energies)
    if output=='save':
        plotter.save(plot_filename)
    if output=='show':
        plotter.show()
    
def plot_vasp_and_wannier90_bandstructure(eigenval_filename,
                                          wannier90hr_filename,poscar_filename,
                                          plot_filename,usedorbitals='all',
                                          usedhoppingcells_rings='all',
                                          output='save'):
    """
    Plot the bandstructure contained in a VASP EIGENVAL file and
    a wannier90_hr.dat file.
    
    usedorbitals: a list of used orbitals to use. Default is 'all'. Note: this only makes
    sense if the selected orbitals don't interact with other orbitals.
    
    usedhoppingcells_rings: If you don't want to use all hopping parameters,
    you can set the number of 'rings' surrounding the main cell here. If it is a list
    (e.g. range(5)), several plots are created.
    The default value is 'all'.
    
    output: determines if the plot is written to a file ('save' - default value) or
    displayed ('show')
    """
    
    #TODO: "ring" is a stupid word
    
    eigenvaldata=eigenval.EigenvalData(eigenval_filename)
    vasp_kpoints=eigenvaldata.kpoints()
    vasp_energies=eigenvaldata.energies()
    
    w90ham=w90hamiltonian.Hamiltonian(wannier90hr_filename,poscar_filename)
    
    #When usedhoppingcells_rings is not a list: create a list with 1 element 
    if not isinstance(usedhoppingcells_rings,list):
        usedhoppingcells_rings=[usedhoppingcells_rings]
        
    for ring in usedhoppingcells_rings:
        if ring=='all':
            cells='all'
        else:
            cells=w90ham.unitcells_within_zone(ring, 'd', numpy.inf)
            
        w90_energies=w90ham.bandstructure_data(vasp_kpoints,'d',
                                               usedhoppingcells=cells,
                                               usedorbitals=usedorbitals)
        plotter=w90hamiltonian.BandstructurePlot()
        
        plotter.plot(vasp_kpoints,vasp_energies,'r-')
        plotter.plot(vasp_kpoints,w90_energies,'b--')
        if output=='save':
            #TODO: dateiname abschneiden, nicht vorstellen
            plotter.save(str(ring)+"_"+plot_filename)
        if output=='show':
            plotter.show()
            
def TopPzBandNrAtGamma(procar_filename,gnrwidth_rings,pzoffset=0):
    """
    Reads a VASP PROCAR file from a zigzag GNR calculation and determines
    the highest \pi* band at the Gamma point (the first point in the PROCAR
    file, actually) that is relevant for the wannier90 calculation.
    
    procar_filename: path to the PROCAR file.
    gnrwidth_rings: number of benzene rings in transverse direction.
    pzoffset (optional): if there are other pz-character bands at the gamma point below
    the \pi* points, the function counts wrong. This value is a manual correction for
    this problem: It assumes that pzoffset more pz-like bands are below the highest
    \pi*-band at the gamma point.
    
    Return:
    highestgoodpzband,energyatgammaofhighestgoodpzband
    
    highestgoodpzband: Band number of the highest pz band at the Gamma point.
    energyatgammaofhighestgoodpzband: Energy of this band at the Gamma point.
    """
    
    procarData=procar.ProcarData(procar_filename)

    chargedata=procarData.chargedata()
    energydata=procarData.energydata()
    
    #Nr of carbon atoms in a GNR of that width
    nrofpzbands=gnrwidth_rings*2+2 
    #Charge data at Gamma point
    gammapointdata=chargedata[0] 
    #Sum the pz charge for a particular band at the gamma point over all ions. Do that for all bands.
    gammapointpzdata=[sum([ion[2] for ion in band]) for band in gammapointdata] 
    #Select band indices where there is pz charge at gamma
    selectpzbands=[i for i in range(len(gammapointpzdata)) if gammapointpzdata[i]>0.] 
    #Get band index of highest pz band at gamma point (index starting with 0, like always)
    highestgoodpzband=selectpzbands[nrofpzbands-1+pzoffset]
    #Energy of that band at gamma point 
    energyatgammaofhighestgoodpzband=energydata[0][highestgoodpzband]
    
    return highestgoodpzband,energyatgammaofhighestgoodpzband

def plot_zigzag_graphene_nanoribbon_pz_bandstructure(wannier90hr_graphene,poscarfile,wannier90woutfile,width,output,usedhoppingcells_rings='all'):
    """
    Plot the \pi bandstructure of a zigzag graphene nanoribbon based on a wannier90 calculation of
    bulk graphene. The \pz orbitals have to be the first two orbitals in the wannier90 file.
    A .dat file with the bandstructure data is also saved as output.dat (each line representing one k-point).
    
    wannier90hr_graphene: path to the wannier90_hr.dat file containing the graphene bulk
    calculation
    poscarfile: path to the VASP POSCAR file of the graphene bulk calculation
    width: width of the ribbon (number of rings).
    output: path to the output image file.
    usedhoppingcells_rings: If you don't want to use all hopping parameters,
    you can set the number of 'rings' surrounding the main cell here. If it is a list
    (e.g. range(5)), several plots are created.
    The default value is 'all'.
    
    Return:
    Hamiltonian: the generated GNR Hamiltonian.
    data: bandstructure data.
    path: path through reciprocal space.
    """
    
    if width%2 == 0:
        unitcells = width/2+1
        get_rid_of=1
    else:
        unitcells = width/2+2
        get_rid_of=3
    
    #When usedhoppingcells_rings is not a list: create a list with 1 element 
    if not isinstance(usedhoppingcells_rings,list):
        usedhoppingcells_rings=[usedhoppingcells_rings]
    
    ham=w90hamiltonian.Hamiltonian.from_file(wannier90hr_graphene,poscarfile,wannier90woutfile)
    
    for ring in usedhoppingcells_rings:    
        if ring=='all':
            cells='all'
        else:
            cells=ham.unitcells_within_zone(ring, 'd', numpy.inf)        
        
        ham2=ham.create_supercell_hamiltonian([[0,0,0],[1,0,0]],[[1,-1,0],[1,1,0],[0,0,1]],usedorbitals=(0,1),usedhoppingcells=cells)
        ham3=ham2.create_supercell_hamiltonian([[0,i,0] for i in range(unitcells)],[[1,0,0],[0,unitcells,0],[0,0,1]])
        ham4=ham3.create_modified_hamiltonian(ham3.drop_dimension_from_cell_list(1),usedorbitals=range(1,ham3.nrorbitals()-get_rid_of))
        path = ham4.point_path([[0,0,0],[0.5,0,0]],100)
        ham4.plot_bandstructure(path,str(ring)+"_"+output,'d')
        data=ham4.bandstructure_data(path, 'd')
        numpy.savetxt(str(ring)+"_"+output+'.dat', numpy.real(data), fmt="%12.6G")
        
        return ham4,data,path
    
def plot_armchair_graphene_nanoribbon_pz_bandstructure(wannier90hr_graphene,poscarfile,wannier90woutfile,width,output,usedhoppingcells_rings='all'):
    """
    Plot the \pi bandstructure of an armchair graphene nanoribbon based on a wannier90 calculation of
    bulk graphene. The \pz orbitals have to be the first two orbitals in the wannier90 file.
    In this example function, width is the number of rings in transversal direction. A .dat file with the bandstructure
    data is also saved as output.dat (each line representing one k-point).
    
    wannier90hr_graphene: path to the wannier90_hr.dat file containing the graphene bulk
    calculation
    poscarfile: path to the VASP POSCAR file of the graphene bulk calculation
    width: width of the ribbon (number of rings). Must be an even number.
    output: path to the output image file.
    usedhoppingcells_rings: If you don't want to use all hopping parameters,
    you can set the number of 'rings' surrounding the main cell here. If it is a list
    (e.g. range(5)), several plots are created.
    The default value is 'all'.
    """
    unitcells = width

    #When usedhoppingcells_rings is not a list: create a list with 1 element 
    if not isinstance(usedhoppingcells_rings,list):
        usedhoppingcells_rings=[usedhoppingcells_rings]

    ham=w90hamiltonian.Hamiltonian.from_file(wannier90hr_graphene,poscarfile,wannier90woutfile)

    for ring in usedhoppingcells_rings:
        if ring=='all':
            cells='all'
        else:
            cells=ham.unitcells_within_zone(ring, 'd', numpy.inf)

        ham2=ham.create_supercell_hamiltonian([[0,0,0],[1,0,0]],[[1,-1,0],[1,1,0],[0,0,1]],usedorbitals=(0,1),usedhoppingcells=cells)
        ham3=ham2.create_supercell_hamiltonian([[i,0,0] for i in range(unitcells)],[[unitcells,0,0],[0,1,0],[0,0,1]])
        ham4=ham3.create_modified_hamiltonian(ham3.drop_dimension_from_cell_list(0))
        path = ham4.point_path([[0,0,0],[0,0.5,0]],100)
        ham4.plot_bandstructure(path,str(ring)+"_"+output,'d')
        data=ham4.bandstructure_data(path, 'd')
        numpy.savetxt(str(ring)+"_"+output+'.dat', numpy.real(data), fmt="%12.6G")

def plot_zigzag_graphene_nanoribbon_pz_bandstructure_nn(nnfile,width,output,magnetic_B=0):
    """
    Plot the \pi bandstructure of a zigzag graphene nanoribbon based on a n-th nearest neighbour parameterization of
    bulk graphene. 
    A .dat file with the bandstructure data is also saved as output.dat (each line representing one k-point).
    
    nnfile: path to the nearest-neighbour input file (see example files)
    width: width of the ribbon (number of rings).
    output: path to the output image file.
    """
    if width%2 == 0:
        unitcells = width/2+1
        get_rid_of=1
    else:
        unitcells = width/2+2
        get_rid_of=3

    ham=w90hamiltonian.Hamiltonian.from_nth_nn_list(nnfile)

    ham2=ham.create_supercell_hamiltonian([[0,0,0],[1,0,0]],[[1,-1,0],[1,1,0],[0,0,1]])
    ham3=ham2.create_supercell_hamiltonian([[0,i,0] for i in range(unitcells)],[[1,0,0],[0,unitcells,0],[0,0,1]])
    ham4=ham3.create_modified_hamiltonian(ham3.drop_dimension_from_cell_list(1),usedorbitals=range(1,ham3.nrorbitals()-get_rid_of),magnetic_B=magnetic_B)
    path = ham4.point_path([[-0.5,0,0],[0.5,0,0]],30)
    ham4.plot_bandstructure(path,output,'d')
    data=ham4.bandstructure_data(path, 'd')
    numpy.savetxt(output+'.dat', numpy.real(data), fmt="%12.6G")

def plot_armchair_graphene_nanoribbon_pz_bandstructure_nn(nnfile,width,output):
    """
    Plot the \pi bandstructure of an armchair graphene nanoribbon based on a n-th nearest neighbour parameterization of
    bulk graphene. 
    In this example function, width is the number of rings in transversal direction. A .dat file with the bandstructure
    data is also saved as output.dat (each line representing one k-point).
    
    nnfile: path to the nearest-neighbour input file (see example files)
    width: width of the ribbon (number of rings). Must be an even number.
    output: path to the output image file.
    """
    unitcells = width
    ham=w90hamiltonian.Hamiltonian.from_nth_nn_list(nnfile)
    ham2=ham.create_supercell_hamiltonian([[0,0,0],[1,0,0]],[[1,-1,0],[1,1,0],[0,0,1]])
    ham3=ham2.create_supercell_hamiltonian([[i,0,0] for i in range(unitcells)],[[unitcells,0,0],[0,1,0],[0,0,1]])
    ham4=ham3.create_modified_hamiltonian(ham3.drop_dimension_from_cell_list(0))
    path = ham4.point_path([[0,-0.5,0],[0,0.5,0]],100)
    ham4.plot_bandstructure(path,output,'d')
    data=ham4.bandstructure_data(path, 'd')
    numpy.savetxt(output+'.dat', numpy.real(data), fmt="%12.6G")