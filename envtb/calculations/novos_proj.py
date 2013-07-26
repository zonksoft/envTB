import envtb.ldos.hamiltonian
import envtb.ldos.potential 
import envtb.ldos.plotter
import envtb.ldos.local_density
import matplotlib.pyplot as plt
import envtb.wannier90.w90hamiltonian as w90hamiltonian
import numpy as np

def LL(N, B):
   
   e = 1.6 * 10**(-19)
   hbar = 1.05 * 10**(-34)
   v_f = 0.82 * 10**6 #0.82* 10**(6)
               
   En = np.zeros(len(B), dtype = float)
   for n in xrange(-N, N + 1): 
      En = np.zeros(len(B), dtype = float)
      for i in xrange(0, len(B)):
         En[i] = np.sign(n) * np.sqrt(2. * e * hbar * v_f**2 * abs(n) * B[i])/1.6 * 10**(19) 
         En1 = np.sign(n) * np.sqrt(2. * e * hbar * v_f**2 * abs(n) * 200)/1.6 * 10**(19) 
      plt.plot(B, En, linewidth=2, color='r')
                                                                                    
   return None


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
    #ham4.plot_bandstructure(path, '' ,'d')
    #plt.ylim(-2, 2)
    #plt.show()
    
    return ham5
# end def define_zigzag_ribbon_w90 

def use_w90(Ny=128, Nx=56, magnetic_B=None):
    
    # Ny: number of atoms in slice is 2*(Ny+1)
    #ham_w90 = define_zigzag_ribbon_w90(
    #    "../../exampledata/02_graphene_3rdnn/graphene3rdnnlist.dat", 
    #    Ny, Nx, magnetic_B=magnetic_B)
    
    #ham = envtb.ldos.hamiltonian.HamiltonianFromW90(ham_w90, Nx)
    
    ham = envtb.ldos.hamiltonian.HamiltonianGraphene(Ny, Nx)
    


    potential = envtb.ldos.potential.Potential2DFromFunction(
        lambda x: 1.4 * np.sin(3.*0.092*(x[1])/2.)**2 +\
                  1.4 * np.sin(3.*0.092*(x[0])/2.)**2)
    ham2 = ham.apply_potential(potential)
    #print ham2.mtot
    envtb.ldos.plotter.Plotter().plot_potential(ham2, ham, alpha=0.7)
    plt.axes().set_aspect('equal')
    plt.show()
    
    #local_dos=envtb.ldos.local_density.LocalDensityOfStates(ham)
    #plt.subplot(2,2,1)
    #envtb.ldos.plotter.Plotter().plot_density(local_dos(0.5), ham.coords)
    ##envtb.ldos.plotter.Plotter().plot_potential(ham2, ham, alpha=0.2)
    #plt.title('E = 0.7')
    #plt.subplot(2,2,2)
    #envtb.ldos.plotter.Plotter().plot_density(local_dos(0.8), ham.coords)
    ##envtb.ldos.plotter.Plotter().plot_potential(ham2, ham, alpha=0.2)
    #plt.title('E = 0.7')
    #plt.subplot(2,2,3)
    #envtb.ldos.plotter.Plotter().plot_density(local_dos(1.0), ham.coords)
    ##envtb.ldos.plotter.Plotter().plot_potential(ham2, ham, alpha=0.2)
    #plt.title('E = 0.7')
    #plt.subplot(2,2,4)
    #envtb.ldos.plotter.Plotter().plot_density(local_dos(1.5), ham.coords)
    ##envtb.ldos.plotter.Plotter().plot_potential(ham2, ham, alpha=0.2)
    #plt.title('E = 0.7')
    ##plt.axes().set_aspect('equal')
    #plt.show()
    
    Barr = np.arange(0., 1000., 2.)
    f_out = open('data_spec.out','w')
    
    sigma_arr = [0., 0.2, 0.5, 0.75, 1.0, 1.2, 1.4, 1.5, 1.9]

    for B in Barr:
        warr = []         
        for sigma in sigma_arr:
           warr = warr + ham2.apply_magnetic_field(magnetic_B=B).eigenvalue_problem(k=150, sigma=sigma)[0].tolist()
        print B, len(warr)
        f_out.writelines(`warr`+'\n')
    
    f_out.close()
    

    #print warr.shape, Barr.shape
    #warr = np.array([ham2.apply_magnetic_field(magnetic_B=B).eigenvalue_problem(k=100, sigma=0.0)
    #                 for B in Barr])
    #[plt.plot(Barr, warr[:,i], 'o', ms=1.5) for i in xrange(warr.shape[1])]
    #LL(4, Barr)
    #plt.show()
    return None
# end def use_w90_example

use_w90()
