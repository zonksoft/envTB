import envtb.ldos.hamiltonian
import numpy as np
import matplotlib.pylab as plt
import envtb.time_propagator.lanczos
import envtb.time_propagator.wave_function
import envtb.time_propagator.vector_potential
import envtb.wannier90.w90hamiltonian as w90hamiltonian

directory = '/tmp/'
#for 1THz pulse (amp=0.5*10**(-2)) use 0.004*10^(-12) and 5500 frames
#for 10THz pulse (amp=0.5*10**(-2)) use 0.001*10^(-12) and 2500 frames
dt = 0.001 * 10**(-12)
NK = 12
laser_freq = 10. * 10**(12)
laser_amp = 0.5 * 10**(-2)
Nc = 3 #number of laser cycles
CEP = np.pi/2.
direct = [np.sqrt(3.)/2.,0.5]
Nframes = 2500
Nx = 200
Ny = 200

def propagate_wave_function(wf_init, hamilt, NK=10, dt=1., maxel=None,
                            num_error=10**(-18), regime='SIL', 
                            file_out=None, **kwrds):
    
    prop = envtb.time_propagator.lanczos.LanczosPropagator(
        wf=wf_init, ham=hamilt, NK=NK, dt=dt)
    
    wf_final, dt_new, NK_new = prop.propagate(
        num_error=num_error, regime=regime)
    
    print 'dt_old = %(dt)g; dt_new = %(dt_new)g; NK_old = %(NK)g; NK_new = %(NK_new)g'\
            % vars()
    print 'norm', wf_final.check_norm()
    
    if file_out is None:
        return wf_final, dt_new, NK_new
    else:
        wf_final.save_wave_function_pic(file_out, maxel, **kwrds)
        return wf_final, dt_new, NK_new

# end def propagate_wave_function

def propagate_graphene_pulse(Nx=20, Ny=20, frame_num=10, magnetic_B=None):
    """
    Since in lanczos in the exponent exp(E*t/hbar) we are using E in eV
    """
    ham = envtb.ldos.hamiltonian.HamiltonianGraphene(Nx, Ny)
    
    w, v = ham.eigenvalue_problem(k=250, sigma=0.0)
    isort = np.argsort(w)
    v = np.array(v)
    wsort = np.sort(w)
    vsort = np.zeros(v.shape, dtype=complex)
    for i in xrange(len(isort)):
        vsort[:,i] = v[:,isort[i]]
    
    #plt.plot(wsort.real, 'o', ms=2)
    
    #plt.show()
    
    ##Nstate = 2
    
    ##envtb.ldos.plotter.Plotter().plot_density(
    ##    vector=abs(vsort[:, Nstate]), coords=ham.coords, alpha=0.7)
    ##plt.show()
    
    ''' Make vector potential'''
    
    Ax = envtb.time_propagator.vector_potential.LP_SinSqEnvelopePulse(
        amplitude_E0=laser_amp, frequency=laser_freq, Nc=Nc, cep=CEP, direction=direct)
    #Ax.plot_pulse()
    #Ax.plot_envelope()
    #Ax.plot_electric_field()
    #plt.show()
    
    for Nstate in xrange(1, 20):
        wf_out = open('wave_functions_%(Nstate)d.out' % vars(), 'w')

        dt_new = dt
        NK_new = NK
        time_counter = 0.0
    
        '''initialize wave function
        create wave function from file (WaveFunction(coords=ham.coords).wave_function_from_file),
        wave function from eigenstate (WaveFunction(vec=v[:, Nstate],coords=ham.coords)) or
        create Gaussian wave packet (GaussianWavePacket(coords=ham.coords, ic=ic, p0=[0.0, 1.5], sigma=7.))
        '''
        #wf_final = envtb.time_propagator.wave_function.WaveFunction(coords=ham.coords)
        #time_counter = wf_final.wave_function_from_file('wave_functions_0.out')
        wf_final = envtb.time_propagator.wave_function.WaveFunction(vec=v[:, Nstate],coords=ham.coords)
        ##ic = Nx/2 * Ny + Ny/2
        ##wf_final = envtb.time_propagator.wave_function.GaussianWavePacket(
        ##        ham.coords, ic, p0=[0.0, 1.5], sigma=7.)
        #maxel = max(wf_final.wf1d)
        
        wf_final.save_wave_function_data(wf_out, time_counter)
    
        import time
    
        '''main loop'''
        for i in xrange(frame_num):
        
            print 'frame %(i)d' % vars()
            time_counter += dt_new
        
            st = time.time()
            ham2 = ham.apply_vector_potential(Ax(time_counter))
            #print 'efficiency ham2', time.time() - st
        
            print 'time', time_counter, 'A', Ax(time)
            st = time.time()
            wf_init = wf_final
            wf_final, dt_new, NK_new = propagate_wave_function(
                  wf_init, ham2, NK=NK_new, dt=dt_new, maxel=None, 
                  regime='TSC', alpha=0.7)
                  #file_out = directory+'f%03d_2d.png' % i)
            #print 'efficiency lanz', time.time() - st
        
            if np.mod(i,10) == 0:
                  wf_final.save_wave_function_data(wf_out, time_counter)
           
        wf_out.close()

propagate_graphene_pulse(Nx=Nx, Ny=Ny, frame_num=Nframes)
