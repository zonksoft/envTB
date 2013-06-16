import envtb.ldos.hamiltonian
import envtb.ldos.potential 
import envtb.ldos.plotter
import envtb.ldos.local_density
import matplotlib.pyplot as plt
import envtb.wannier90.w90hamiltonian as w90hamiltonian
import numpy as np

def define_zigzag_ribbon_w90(nnfile, width, length, magnetic_B=None):
    
    if width%2 == 0:
        unitcells = width/2 + 1
        get_rid_of = 1
    else:
        unitcells = width/2 + 2
        get_rid_of = 3

    #ham = w90hamiltonian.Hamiltonian.from_nth_nn_list(nnfile)
    ham = w90hamiltonian.Hamiltonian.from_file("../../exampledata/01_graphene_vasp_wannier90/wannier90_hr.dat",
                                               "../../exampledata/01_graphene_vasp_wannier90/POSCAR",
                                               "../../exampledata/01_graphene_vasp_wannier90/wannier90.wout",
                                               "../../exampledata/01_graphene_vasp_wannier90/OUTCAR")
    print ham.maincell_hamiltonian_matrix()
    ham2 = ham.create_supercell_hamiltonian(
        [[0, 0, 0], [1, 0, 0]],
        [[1, -1, 0], [1, 1, 0], [0, 0, 1]],
        usedorbitals=(0, 1))
    
    ham3 = ham2.create_supercell_hamiltonian(
        [[0, i, 0] for i in range(unitcells)], 
        [[1, 0, 0], [0, unitcells, 0], [0, 0, 1]])
    
    ham4 = ham3.create_modified_hamiltonian(
        ham3.drop_dimension_from_cell_list(1),
        usedorbitals=range(1, ham3.nrorbitals()-get_rid_of),
        magnetic_B=magnetic_B)
       
    ham5 = ham4.create_supercell_hamiltonian(
        [[i, 0, 0] for i in range(length)], 
        [[length, 0, 0], [0, 1, 0], [0, 0, 1]],
        output_maincell_only=True)
 
    ham6 = ham5.create_modified_hamiltonian(
        usedorbitals=range(1, ham5.nrorbitals()-1))
    
    path = ham4.point_path([[-0.5,0,0],[1.0,0,0]],300)
    ham4.plot_bandstructure(path, '' ,'d')
    plt.ylim(-2, 2)
    plt.show()
    
    return ham5
# end def define_zigzag_ribbon_w90 
    
def use_w90(Ny=43, Nx=40, magnetic_B=None):
    
    # Ny: number of atoms in slice is 2*(Ny+1)
    ham_w90 = define_zigzag_ribbon_w90(
        "../../exampledata/02_graphene_3rdnn/graphene3rdnnlist.dat", 
        Ny, Nx, magnetic_B=magnetic_B)
    
    ham = envtb.ldos.hamiltonian.HamiltonianFromW90(ham_w90, Nx)
    
    potential = envtb.ldos.potential.Potential2DFromFunction(
        lambda x: 0.1 * np.sin(0.25*(x[1])/2.)**2 +\
                  0.1 * np.sin(0.25*(x[0])/2.)**2)
    
    ham2 = ham.apply_potential(potential)
    envtb.ldos.plotter.Plotter().plot_potential(ham2, ham)
    plt.axes().set_aspect('equal')
    plt.show()
    
    local_dos=envtb.ldos.local_density.LocalDensityOfStates(ham)
    
    envtb.ldos.plotter.Plotter().plot_density(local_dos(0.7), ham.coords)
    plt.title('E = 0.7')
    plt.axes().set_aspect('equal')
    plt.show()
    
    return None
# end def use_w90_example

use_w90()