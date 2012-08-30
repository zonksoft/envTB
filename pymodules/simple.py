from vasp import eigenval
from wannier90 import w90hamiltonian
import numpy

"""
This file contains functions for often used procedures.
They can also be considered as use cases.
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
                                          wannier90hr_filename,poscar_filename,plot_filename):
    """
    Plot the bandstructure contained in a VASP EIGENVAL file and
    a wannier90_hr.dat file.
    """
    
    eigenvaldata=eigenval.EigenvalData(eigenval_filename)
    vasp_kpoints=eigenvaldata.kpoints()
    vasp_energies=eigenvaldata.energies()
    
    w90ham=w90hamiltonian.Hamiltonian(wannier90hr_filename,poscar_filename)
    w90_energies=w90ham.bandstructure_data(vasp_kpoints,'d')
    plotter=w90hamiltonian.BandstructurePlot()
    
    plotter.plot(vasp_kpoints,vasp_energies,'r-')
    plotter.plot(vasp_kpoints,w90_energies,'b.')
    plotter.save(plot_filename)