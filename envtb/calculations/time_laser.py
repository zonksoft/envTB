import envtb.ldos.hamiltonian
import envtb.ldos.potential
import envtb.ldos.plotter
import numpy as np
import matplotlib.pylab as plt
import envtb.time_propagator.lanczos
import envtb.time_propagator.wave_function
import envtb.wannier90.w90hamiltonian as w90hamiltonian
import envtb.time_propagator.vector_potential
import envtb.ldos.plotter
import envtb.time_propagator.current

directory = '/tmp/'
dt = 0.1 * 10**(-15)
NK = 12
laser_freq = 10**(12)
laser_amp = 10.0 * 10**(-2)
Nc = 2

def propagate_wave_function(wf_init, hamilt, NK=10, dt=1., maxel=None,
                            num_error=10**(-18), regime='SIL', 
                            file_out='/tmp/a.png'):
    
    prop = envtb.time_propagator.lanczos.LanczosPropagator(
        wf=wf_init, ham=hamilt, NK=NK, dt=dt)
    
    wf_final, dt_new, NK_new = prop.propagate(
        num_error=num_error, regime=regime)
    
    print 'dt_old = %(dt)g; dt_new = %(dt_new)g; NK_old = %(NK)g; NK_new = %(NK_new)g'\
            % vars()
    print 'norm', wf_final.check_norm()
    
    wf_final.plot_wave_function(maxel)
    plt.axes().set_aspect('equal')
    #plt.ylim(-100,100)
    plt.savefig(file_out)
    plt.close()
    
    #envtb.ldos.plotter.Plotter(xval=range(len(wf_final.wf1d)), yval=np.abs(wf_final.wf1d)).plotting()
    #plt.savefig(file_out+'_new.png')
    
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

def propagate_graphene_flatpulse(Nx=50, Ny=50, frame_num=1500):
    
    """
    Since in lanczos in the exponent exp(E*t/hbar) we are using E in eV    
   
    """
    ham = envtb.ldos.hamiltonian.HamiltonianGraphene(Ny, Nx)
    #ham = envtb.ldos.hamiltonian.HamiltonianTB(Ny, Nx)
    #ham = ham1.make_periodic_y()
    #ham = ham1.make_periodic_x()    
    
    ic = Nx/2 * Ny + Ny/2
    
    Ax = envtb.time_propagator.vector_potential.LP_SinSqEnvelopePulse(
        amplitude_E0 = laser_amp, frequency = laser_freq, Nc = Nc)
    #envtb.ldos.plotter.Plotter().plot_density(np.abs(v[:,im]), ham.coords)
    
    wf_final = wf_init_gaussian_wave_packet(
        ham.coords, ic, p0=[0.0, 1.5], sigma=7.)
    plt.close()

    dt_new = dt
    NK_new = NK
    
    wf_out = open('wave_functions.out','w')
    dt_new = dt
    NK_new = NK
    time = 0.0
    
    maxel = 0.7*max(wf_final.wf1d)
    wf_out.writelines(`time`+'   '+`wf_final.wf1d.tolist()`+'\n')
    
    for i in xrange(100):
        print 'frame %(i)d' % vars()
        time += dt_new
        import time
        st = time.time()
        ham2 = ham.apply_vector_potential(Ax(time))
        print time.time()-st
        #print ham2.mtot
        
        wf_init = wf_final
        wf_final, dt_new, NK_new = propagate_wave_function(
            wf_init, ham2, NK=NK_new, dt=dt_new, 
            maxel = maxel, regime='TSC', 
            file_out = directory+'f%03d_2d.png' % i)
        
        if np.mod(i,1) == 0:
            #tar.append(time)
            wf_out.writelines(`time`+'   '+`wf_final.wf1d.tolist()`+'\n')
           
    wf_out.close()
        
propagate_graphene_flatpulse()
