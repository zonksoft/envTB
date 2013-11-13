#! /usr/bin/env python
import envtb.ldos.hamiltonian
import numpy as np
#import matplotlib.pylab as plt
import envtb.time_propagator.lanczos
import envtb.time_propagator.wave_function
import envtb.time_propagator.vector_potential
import envtb.wannier90.w90hamiltonian as w90hamiltonian

##directory = '/tmp/'
##for 1THz pulse (amp=0.5*10**(-2)) use 0.004*10^(-12) and 5500 frames
##for 10THz pulse (amp=0.5*10**(-2)) use 0.001*10^(-12) and 2500 frames
dt = 0.001 * 10**(-12)
print dt
NK = 12
laser_freq = 10. * 10**(12)
laser_amp = 0.5 * 10**(-2)
Nc = 3 #number of laser cycles
CEP = np.pi/2.
direct = [1.0, 0.0] #[np.sqrt(3.)/2.,0.5]
Nframes = 2500
Nx = 300
Ny = 300

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

    Nall = 250

    w, v = ham.sorted_eigenvalue_problem(k=Nall, sigma=0.0)

    '''
        Store eigenvalue_problem
    '''
    fout = open('eigenvalue_problem.out', 'w')
    for i in xrange(Nall):
        fout.writelines(`w[i]`+'   '+`v[:,i].tolist()`+'\n')


    ''' Make vector potential'''

    A_pot = envtb.time_propagator.vector_potential.SinSqEnvelopePulse(
        amplitude_E0=laser_amp, frequency=laser_freq, Nc=Nc, cep=CEP, direction=direct)

    import pypar

    proc = pypar.size()                                # Number of processes as specified by mpirun
    myid = pypar.rank()                                # Id of of this process (myid in [0, proc-1]) 
    node = pypar.get_processor_name()                  # Host name on which current process is running

    Nthread = Nall / proc

    N_range = range(myid * Nthread, (myid + 1) * Nthread, 10)

    for Nstate in N_range:

        wf_out = open('wave_functions_%(Nstate)d.out' % vars(), 'w')
        expansion_out = open('expansion_%(Nstate)d.out' % vars(), 'w')
        coords_out = open('coords_current_%(Nstate)d.out' % vars(), 'w')
        dipole_out = open('dipole_%(Nstate)d.out' % vars(), 'w')

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

            #print 'frame %(i)d' % vars()
            time_counter += dt_new

            st = time.time()
            ham2 = ham.apply_vector_potential(A_pot(time_counter))
            #print 'efficiency ham2', time.time() - st

            #print 'time', time_counter, 'A', A_pot(time)
            st = time.time()
            wf_init = wf_final
            wf_final, dt_new, NK_new = propagate_wave_function(
                  wf_init, ham2, NK=NK_new, dt=dt_new, maxel=None,
                  regime='TSC', alpha=0.7)
                  #file_out = directory+'f%03d_2d.png' % i)
            #print 'efficiency lanz', time.time() - st

            if np.mod(i,10) == 0:
                  wf_final.save_wave_function_data(wf_out, time_counter)
                  wf_final.save_wave_function_expansion(expansion_out, v)
                  wf_final.save_coords_current(coords_out, A_pot(time))


        wf_out.close()
        expansion_out.close()
        coords_out.close()
        dipole_out.close()

    pypar.finalize()

    return None

propagate_graphene_pulse(Nx=Nx, Ny=Ny, frame_num=Nframes)
