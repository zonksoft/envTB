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

"""TODO: Add function to view and zoom"""

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
    
    #nrbands,nrkpoints,nrions=procarData.info()
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