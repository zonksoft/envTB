import envtb.ldos.hamiltonian
import envtb.ldos.potential
import envtb.ldos.plotter
import numpy as np
import matplotlib.pylab as plt
import envtb.time_propagator.lanczos
import envtb.time_propagator.wave_function
import envtb.wannier90.w90hamiltonian as w90hamiltonian

directory = '/tmp/'

def propagate_wave_function(wf_init, hamilt, NK=10, dt=1., maxel=None,
                            num_error=10**(-18), regime='SIL', 
                            file_out='/tmp/a.png'):
    
    prop = envtb.time_propagator.lanczos.LanczosPropagator(
        wf=wf_init, ham=hamilt, NK=NK, dt=dt)
    
    wf_final, dt_new, NK_new = prop.propagate(
        num_error=num_error, regime=regime)
    
    print 'dt_old = %(dt)g; dt_new = %(dt_new)g; NK_old = %(NK)g; NK_new = %(NK_new)g' % vars()
    print 'norm', wf_final.check_norm()  #np.dot(np.transpose(np.conjugate(wf_final)), wf_final)
    
    wf_final.plot_wave_function(maxel)
    plt.axes().set_aspect('equal')
    plt.savefig(file_out)
    plt.close()

    return wf_final, dt_new, NK_new

def wf_init_from_electron_density(hamilt, mu=0.1, kT=0.0025):

    wf0 = envtb.time_propagator.wave_function.WaveFunction0(hamilt, mu, kT)

    dens1 = hamilt.electron_density(mu, kT)

    envtb.ldos.plotter.Plotter().plot_density(
        vector=dens1, coords=hamilt.coords)

    plt.show()

    wf0.plot_wave_function()
    return wf0

def wf_init_gaussian_wave_packet(coords, ic, p0=[0.0, 0.0], sigma=5.):
    
    wf0 = envtb.time_propagator.wave_function.GaussianWavePacket(
        coords=coords, ic=ic, p0=p0, sigma=sigma)
    
    wf0.plot_wave_function()
    
    return wf0

def propagate_gauss_TB(Nx=50, Ny=40, frame_num=100):
    
    ham = envtb.ldos.hamiltonian.HamiltonianTB(Ny, Nx)
    
    ic = Nx/2 * Ny + Ny/2 + 2
    
    wf_final = wf_init_gaussian_wave_packet(ham.coords, ic, sigma=10.)
    
    maxel = 0.8 * np.max(wf_final.wf1d)
    wf_final.plot_wave_function(maxel)
    plt.axes().set_aspect('equal')
    plt.savefig(directory+'0%d_2d.png' % 0)
    plt.close()

    for i in xrange(frame_num):
    
        wf_init = wf_final
        wf_final = propagate_wave_function(wf_init, ham, maxel = maxel, 
            file_out = directory+'f%03d_2d.png' % i)[0]
    
    return None
    
def propagate_el_den_TB(Nx=50, Ny=40, mu=0.05, frame_num=100):
    
    ham = envtb.ldos.hamiltonian.HamiltonianTB(Ny, Nx)
    
    wf_final = wf_init_from_electron_density(ham, mu = mu)
    
    maxel =  1.2 * np.max(wf_final.wf1d)
    wf_final.plot_wave_function(maxel)
    plt.axes().set_aspect('equal')
    plt.savefig('../../../../Desktop/pics_2d/TB/0%d_2d.png' % 0)
    plt.close()
                    
    for i in xrange(frame_num):
        
        potential = envtb.ldos.potential.Potential1DFromFunction(
            lambda x: -5. * (Ny/2 - x) * 2. / Ny * np.sin(0.1 * i))
        
        ham2 = ham.apply_potential(potential)
        
        envtb.ldos.plotter.Plotter().plot_potential(
            ham2, ham, maxel = 5, minel = -5)
        
        plt.axes().set_aspect('equal')
        plt.savefig('../../../../Desktop/pics_2d/TB/pot%03d_2d.png' % i)
        plt.close()
        
        
        wf_init = wf_final
        wf_final = propagate_wave_function(wf_init, ham2, maxel = maxel, 
            file_out = '../../../../Desktop/pics_2d/TB/f%03d_2d.png' % i)
    
    return None
    
def propagate_gauss_graphene(Nx=30, Ny=30, frame_num=100):
    
    ham = envtb.ldos.hamiltonian.HamiltonianGraphene(Ny, Nx)
    
    ic = Nx/2 * Ny + Ny/2 + 2
    
    wf_final = wf_init_gaussian_wave_packet(
        ham.coords, ic, p0=[1.0, 0.0], sigma=7.)
    
    maxel = 0.8 * np.max(wf_final.wf1d)
    wf_final.plot_wave_function(maxel)
    plt.axes().set_aspect('equal')
    plt.savefig('../../../../Desktop/pics_2d/graphene/0%d_2d.png' % 0)
    plt.close()
    
    dt_new = 0.5
    NK_new = 12
    
    for i in xrange(frame_num):
        print 'frame %(i)d' % vars()
        wf_init = wf_final
        wf_final, dt_new, NK_new = propagate_wave_function(
            wf_init, ham, NK=NK_new, dt=dt_new, maxel = maxel, regime='SIL', 
            file_out = '../../../../Desktop/pics_2d/graphene/f%03d_2d.png'
            % i)
     
    return None
    
def propagate_el_den_graphene(Nx=50, Ny=40, mu=0.01, frame_num=200):
    
    ham = envtb.ldos.hamiltonian.HamiltonianGraphene(Ny, Nx)
    
    wf_final = wf_init_from_electron_density(ham, mu = mu)
    
    maxel = 0.8 * np.max(wf_final.wf1d)
    wf_final.plot_wave_function(maxel)
    plt.axes().set_aspect('equal')
    plt.savefig('../../../../Desktop/pics_2d/graphene_el_den/0%d_2d.png' % 0)
    plt.close()
        
    for i in xrange(frame_num):
        
        potential = envtb.ldos.potential.Potential1DFromFunction(
            lambda x: -10. * (Ny/2 - x) * 2. / Ny * np.sin(0.05 * i))
        ham2 = ham.apply_potential(potential)
        
        envtb.ldos.plotter.Plotter().plot_potential(
            ham2, ham, maxel = 10.1, minel = -10.1)
        
        plt.axes().set_aspect('equal')
        plt.savefig(
            '../../../../Desktop/pics_2d/graphene_el_den/pot%03d_2d.png' % i)
        plt.close()
        
        wf_init = wf_final
        wf_final = propagate_wave_function(wf_init, ham, maxel = maxel, 
            file_out = '../../../../Desktop/pics_2d/graphene_el_den/f%03d_2d.png'
            % i)[0]

    return None

def define_zigzag_ribbon_w90(nnfile, width, length, magnetic_B=None):
    if width%2 == 0:
        unitcells = width/2+1
        get_rid_of=1
    else:
        unitcells = width/2+2
        get_rid_of=3

    ham=w90hamiltonian.Hamiltonian.from_nth_nn_list(nnfile)
    
    ham2=ham.create_supercell_hamiltonian(
        [[0,0,0],[1,0,0]], [[1,-1,0],[1,1,0],[0,0,1]])
    
    ham3=ham2.create_supercell_hamiltonian(
        [[0,i,0] for i in range(unitcells)],
        [[1,0,0],[0,unitcells,0],[0,0,1]])
    
    ham4=ham3.create_modified_hamiltonian(
        ham3.drop_dimension_from_cell_list(1),
        usedorbitals=range(1,ham3.nrorbitals()-get_rid_of),
        magnetic_B=magnetic_B)
       
    ham5=ham4.create_supercell_hamiltonian(
        [[i,0,0] for i in range(length)], [[length,0,0],[0,1,0],[0,0,1]],
        output_maincell_only=True)
    
    path = ham4.point_path([[-0.5,0,0],[1.0,0,0]],100)
    ham4.plot_bandstructure(path, '' ,'d')
    plt.ylim(-2, 2)
    plt.show()
    return ham5
        
def propagate_gauss_graphene_W90(Nx=30, Ny=20, magnetic_B=None,
                                 frame_num=100):
    
    ham_w90 = define_zigzag_ribbon_w90(
        "../../exampledata/02_graphene_3rdnn/graphene3rdnnlist.dat", 
        Ny, Nx, magnetic_B=magnetic_B)
    
    ham = envtb.ldos.hamiltonian.HamiltonianFromW90(ham_w90, Nx)
    
    ic = ham.Nx/2 * ham.Ny + ham.Ny/2 + 2 
    
    wf_final = wf_init_gaussian_wave_packet(ham.coords, ic, sigma = 10.)
    
    maxel = 0.8 * np.max(wf_final.wf1d)
    wf_final.plot_wave_function(maxel)
    plt.axes().set_aspect('equal')
    plt.savefig('../../../../Desktop/pics_2d/grapheneW90/0%d_2d.png' % 0)
    plt.close()

    for i in xrange(frame_num):
    
        wf_init = wf_final
        wf_final = propagate_wave_function(wf_init, ham, maxel=maxel,
            file_out='../../../../Desktop/pics_2d/grapheneW90/f%03d_2d.png' % i)[0]
    
    return None
    
def propagate_el_den_graphene_W90(Nx=50, Ny=40, magnetic_B=None, mu=0.01, 
                                  frame_num=100):
    
    ham_w90 = define_zigzag_ribbon_w90("../../exampledata/02_graphene_3rdnn/graphene3rdnnlist.dat", 
                                       Ny, Nx, magnetic_B=magnetic_B)
    
    ham = envtb.ldos.hamiltonian.HamiltonianFromW90(ham_w90, Nx)
    
    wf_final = wf_init_from_electron_density(ham, mu=mu)
    
    maxel = 0.8 * np.max(wf_final.wf1d)
    wf_final.plot_wave_function(maxel)
    plt.axes().set_aspect('equal')
    plt.savefig('../../../../Desktop/pics_2d/grapheneW90_el_den/0%d_2d.png' % 0)
    plt.close()

    for i in xrange(frame_num):
    
        wf_init = wf_final
        wf_final = propagate_wave_function(wf_init, ham, maxel=maxel, 
            file_out='../../../../Desktop/pics_2d/grapheneW90_el_den/f%03d_2d.png'
            % i)[0]
    
    return None
    
    
propagate_gauss_TB()
#propagate_el_den_TB()

#propagate_gauss_graphene()
#propagate_el_den_graphene()
#propagate_gauss_graphene_W90()
#propagate_el_den_graphene_W90()

