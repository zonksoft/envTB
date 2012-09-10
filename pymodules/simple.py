from vasp import eigenval
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

def plot_vasp_bandstructure(eigenval_filename,plot_filename):
    """
    Plot the bandstructure contained in a VASP EIGENVAL file.
    The output format is determined by the file ending of
    plot_filename.
    """
    
    eigenvaldata=eigenval.EigenvalData(eigenval_filename)
    kpoints=eigenvaldata.kpoints()
    energies=eigenvaldata.energies()
    plotter=w90hamiltonian.BandstructurePlot()
    plotter.plot(kpoints,energies)
    plotter.save(plot_filename)
    
def plot_vasp_and_wannier90_bandstructure(eigenval_filename,
                                          wannier90hr_filename,poscar_filename,
                                          plot_filename,usedorbitals='all',
                                          usedhoppingcells_rings='all'):
    """
    Plot the bandstructure contained in a VASP EIGENVAL file and
    a wannier90_hr.dat file.
    
    usedorbitals: a list of used orbitals to use. Default is 'all'. Note: this only makes
    sense if the selected orbitals don't interact with other orbitals.
    
    usedhoppingcells_rings: If you don't want to use all hopping parameters,
    you can set the number of 'rings' surrounding the main cell here. If it is a list
    (e.g. range(5)), several plots are created.
    The default value is 'all'.
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
        #TODO: dateiname abschneiden, nicht vorstellen
        plotter.save(str(ring)+"_"+plot_filename)