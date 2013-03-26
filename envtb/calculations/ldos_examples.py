import envtb.ldos.hamiltonian
import envtb.ldos.potential 
import envtb.ldos.plotter
import envtb.ldos.local_density
import matplotlib.pyplot as plt
import envtb.wannier90.w90hamiltonian as w90hamiltonian
import numpy as np

def electron_density_example(Nx = 20, Ny = 20, mu = 0.5, kT = 0.0025):
    ####################ELECTRON_DENSITY######################################
    ham = envtb.ldos.hamiltonian.HamiltonianTB(Ny, Nx)
    ham2 = ham.make_periodic_y()
    ham3 = ham2.make_periodic_x()
        
    dens = ham3.electron_density(mu, kT)
        
    envtb.ldos.plotter.Plotter(range(ham.Ny), dens[ham.Ny:2*ham.Ny]).plotting()
    plt.show()
    
    envtb.ldos.plotter.Plotter().plot_density(dens, ham.coords)
    
    plt.show()
    ########################END###############################################
    return None

def electron_density_graphene_example(Nx=50, Ny=30, mu=0.0, kT=0.0025):
    ######################ELECTRON_DENSITY_GRAPHENE###########################
    ham=envtb.ldos.hamiltonian.HamiltonianGraphene(Ny, Nx)
    ham2 = ham.make_periodic_x()
    #ham.make_periodic_y()
    #ham.apply_potential(potential)
    dens1 = ham2.electron_density(mu, kT)
    #dens2 = ham.electron_density(0, kT, '2d')
    #envtb.ldos.plotter.Plotter().plot_density(Nx, Ny, dens1-dens2, 'graphene')
    envtb.ldos.plotter.Plotter().plot_density(dens1, ham.coords)
    plt.show()
    ##########################################################################
    return None

def plot_ldos_example(Nx=20, Ny=30):
    #####################LOCAL_DENSITY_OF_STATES##############################
    potential = envtb.ldos.potential.Potential1DFromFunction(
        lambda x: - 5. * (Ny/2-x) * 2 / Ny)
    ham = envtb.ldos.hamiltonian.HamiltonianTB(Ny, Nx)
    
    ham2 = ham.apply_potential(potential)
    
    envtb.ldos.plotter.Plotter().plot_potential(ham2, ham)
    plt.axes().set_aspect
    plt.show()
    
    local_dos=envtb.ldos.local_density.LocalDensityOfStates(ham2)
    
    plt.subplot(2,2,1)
    envtb.ldos.plotter.Plotter().plot_density(local_dos(0.01), ham2.coords)
    plt.title('E = 0.01')
    plt.subplot(2,2,2)
    envtb.ldos.plotter.Plotter().plot_density(local_dos(0.2), ham2.coords)
    plt.title('E = 0.2')
    plt.subplot(2,2,3)
    envtb.ldos.plotter.Plotter().plot_density(local_dos(0.5), ham2.coords)
    plt.title('E = 0.5')
    plt.subplot(2,2,4)
    envtb.ldos.plotter.Plotter().plot_density(local_dos(1.0), ham2.coords)
    plt.title('E = 1.0')
    plt.show()
    
    envtb.ldos.plotter.Plotter().plot_density(local_dos(0.01), ham2.coords)
    plt.axes().set_aspect('equal')
    plt.show()
    ########################END###############################################
    return None

def plot_ldos_graphene_example(Nx=50, Ny=20):
    #####################LOCAL_DENSITY_OF_STATES_GRAPHENE#####################
    ham = envtb.ldos.hamiltonian.HamiltonianGraphene(Ny, Nx)
    #ham.mtot[2359,2359] = 10 # introduce defect!!!
    #ham.make_periodic_y()
    local_dos = envtb.ldos.local_density.LocalDensityOfStates(ham)
  
    plt.subplot(2,2,1)
    envtb.ldos.plotter.Plotter().plot_density(local_dos(0.0), ham.coords)
    plt.title('E = 0.0')
    plt.subplot(2,2,2)
    envtb.ldos.plotter.Plotter().plot_density(local_dos(0.15), ham.coords)
    plt.title('E = 0.15')
    plt.subplot(2,2,3)
    envtb.ldos.plotter.Plotter().plot_density(local_dos(0.5), ham.coords)
    plt.title('E = 0.5')
    plt.subplot(2,2,4)
    envtb.ldos.plotter.Plotter().plot_density(local_dos(1.2), ham.coords)
    plt.title('E = 1.2')
    plt.show()
    ########################END###############################################
    return None

def plot_ldos_example_2Dpot(Nx=30, Ny=40):
    #####################LOCAL_DENSITY_OF_STATES##############################
    ham=envtb.ldos.hamiltonian.HamiltonianTB(Ny, Nx)
    
    i0 = ham.Nx / 2
    j0 = ham.Ny / 2
    ic = i0*ham.Ny + j0
    potential = envtb.ldos.potential.Potential2DFromFunction(
        lambda x: 0.01 * (ham.coords[ic][1]-x[1])**2 + 
                  0.01 * (ham.coords[ic][0]-x[0])**2)
    
    ham2 = ham.apply_potential(potential)
    #ham.make_periodic_y()
    local_dos=envtb.ldos.local_density.LocalDensityOfStates(ham2)
        
    envtb.ldos.plotter.Plotter().plot_density(local_dos(1.5), ham2.coords)
    plt.axes().set_aspect('equal')
    plt.show()
    ########################END###############################################
    return None

def define_zigzag_ribbon_w90(nnfile, width, length, magnetic_B=None):
    
    if width%2 == 0:
        unitcells = width/2 + 1
        get_rid_of = 1
    else:
        unitcells = width/2 + 2
        get_rid_of = 3

    ham = w90hamiltonian.Hamiltonian.from_nth_nn_list(nnfile)
    
    ham2 = ham.create_supercell_hamiltonian(
        [[0, 0, 0], [1, 0, 0]],
        [[1, -1, 0], [1, 1, 0], [0, 0, 1]])
    
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
    
    path = ham4.point_path([[-0.5,0,0],[1.0,0,0]],100)
    ham4.plot_bandstructure(path, '' ,'d')
    plt.ylim(-2, 2)
    plt.show()
    
    return ham6
    
def use_w90_example(Ny=30, Nx=40, magnetic_B=None):
    ########################USE_W90###########################################
    # Attention!   Ny: number of atoms in slice is 2*(Ny+1)
    ham_w90 = define_zigzag_ribbon_w90(
        "../../exampledata/02_graphene_3rdnn/graphene3rdnnlist.dat", 
        Ny, Nx, magnetic_B=magnetic_B)
    
    ham = envtb.ldos.hamiltonian.HamiltonianFromW90(ham_w90, Nx)
   
    i0 = ham.Nx / 2
    j0 = ham.Ny / 2
    ic = (i0 - 1) * ham.Ny + (j0-1)
    #print ic, ham.Ntot, ham.Ny
    potential = envtb.ldos.potential.Potential2DFromFunction(
        lambda x: 0.01 * (ham.coords[ic][1] - x[1])**2 + 0.01 * \
                 (ham.coords[ic][0] - x[0])**2)
    
    ham2 = ham.apply_potential(potential)
    envtb.ldos.plotter.Plotter().plot_potential(ham2, ham)
    
    plt.axes().set_aspect('equal')
    plt.show()
   
    local_dos=envtb.ldos.local_density.LocalDensityOfStates(ham)
    
    envtb.ldos.plotter.Plotter().plot_density(local_dos(0.7), ham.coords)
    plt.title('E = 0.7')
    plt.axes().set_aspect('equal')
    plt.show()
    ##########################################################################
    return None

#use_w90_example(magnetic_B=0)
#plot_ldos_example()
electron_density_example()
electron_density_graphene_example()
plot_ldos_graphene_example()
plot_ldos_example_2Dpot()

