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

def make_fourier(wfinal, Nx, Ny):
    
    wf_f2D = np.resize(wfinal.wf1d, (Nx, Ny))
    wfur1 = np.fft.fft2(wf_f2D)
    #wfur2 = np.zeros([Nx, Ny], dtype = complex)
    #wfur2[:Nx/2, :Ny/2] = wfur1[Nx/2:, Ny/2:]
    #wfur2[Nx/2:, Ny/2:] = wfur1[:Nx/2, :Ny/2]
    #wfur2[:Nx/2, Ny/2:] = wfur1[Nx/2:, :Ny/2]
    #wfur2[Nx/2:, :Ny/2] = wfur1[:Nx/2, Ny/2:]
    
    return wfur1.T

def propagate_graphene_flatpulse(Nx=30, Ny=30, frame_num=1500):
    """
    Since in lanczos in the exponent exp(E*t/hbar) we are using E in eV    
   
    """
    ham = envtb.ldos.hamiltonian.HamiltonianGraphene(Ny, Nx)
    #ham = envtb.ldos.hamiltonian.HamiltonianTB(Ny, Nx)
    #ham = ham1.make_periodic_y()
    #ham = ham1.make_periodic_x()    
    ic = Nx/2 * Ny + Ny/2
    w, v = ham.eigenvalue_problem()
    im = np.where(w==np.max(w))[0][0]
    
    #envtb.ldos.plotter.Plotter().plot_density(np.abs(v[:,im]), ham.coords)
                   
    wf_final = wf_init_gaussian_wave_packet(
        ham.coords, ic, p0=[1.0, 0.0], sigma=5.)
    plt.close()
    
    #wf_final.wf1d[::2] = wf_final.wf1d[::2]*np.complex(1., 0.0)
    #plt.plot(wf_final.wf1d)
    #plt.show()

    plt.subplot(2, 2, 1)
    plt.plot(w, 'o', ms=2)
    plt.subplot(2, 2, 2)
    a_n = np.array([np.dot(np.conjugate(v[:, i]), wf_final.wf1d) for i in xrange(len(w))])
    # envtb.ldos.plotter.Plotter().plot_density(vector=a_n, coords=wf_final.coords)
    plt.plot(w, np.abs(a_n), 'o', ms=2)
    plt.subplot(2, 2, 3)
    ham.plot_bandstructure()

    plt.show()

    current = envtb.time_propagator.current.CurrentOperator()
    J = current(wf_final)
    print "current", J
    #wf_final = wf_init_from_electron_density(hamilt=ham, mu=0.1, kT=0.0025)
   
        
    maxel = 0.5 * np.max(wf_final.wf1d)
       
    wf_final.plot_wave_function(maxel)
    plt.axes().set_aspect('equal')
    #plt.ylim(-100,100)
    plt.savefig('../../../../Desktop/pics_2d/TB2_laser/0%d_2d.png' % 0)
    plt.close()
    
    try:
        wfur = make_fourier(wf_final, Nx, Ny)
        plt.pcolor(abs(wfur))
        plt.xlabel(r'$k_x$')
        plt.ylabel(r'$k_y$')
        plt.colorbar()
    except:
        plt.plot(abs(np.fft.fft(wf_final.wf1d)))
    plt.savefig('../../../../Desktop/pics_2d/TB2_laser/fur_0.png')
    plt.close()
        
    plt.plot(abs(wf_final.wf1d))
    plt.savefig('../../../../Desktop/pics_2d/TB2_laser/wf1D_0.png')
    plt.close()
    
    dt_new = 0.5 * 10**(-15)
    NK_new = 12
    
    #Ax = envtb.time_propagator.vector_potential.LP_FlatTopPulse(
    #    amplitude_E0 = 2.6 * 10**(6), frequency = 10**12)
    #Ax.plot_pulse()
    #ham_with_vec_pot = ham.apply_vector_potential(Ax(1* 10**(-12)))
    #Ax.plot_pulse()
    
    Ax = envtb.time_propagator.vector_potential.LP_SinSqEnvelopePulse(
        amplitude_E0 = 0 * 10**(2), frequency = 1 * 10**(16), Nc = 2)
    
    ###Ax = envtb.time_propagator.vector_potential.VectorPotentialWave(
    ###    amplitude_E0 = 2 * 10**(2), frequency = 1. * 10**(16))
    Ax.plot_pulse()
    #Ax.plot_envelope()
    Ax.plot_electric_field()   
    plt.show()
    
    #ham_with_mag = ham.apply_magnetic_field(magnetic_B=500)
    #ham_with_mag.plot_bandstructure()
    xav = []
    yav = []
    time_ar = []
    U = []    
    time = 0
    
    a = 1.42    #For graphene
    
    wf = []
    
    
    
    for i in xrange(frame_num):
        print 'frame %(i)d' % vars()
        time += dt_new
        ham2 = ham.apply_vector_potential(Ax(time))
        #potential = envtb.ldos.potential.Potential1DFromFunction(
        #    lambda x: -5000. * x / Ny * np.cos(0.05 * i))
        #ham2 = ham.apply_potential(potential)
        #ham_with_vec_pot = ham.dipole_approximation(Ax, dt_new)
        
        envtb.ldos.plotter.Plotter().plot_potential(ham_mit_pot=ham2, ham_bare=ham, maxel=100.1, minel=-100.1)
        plt.savefig('../../../../Desktop/pics_2d/TB2_laser/pot_%d.png' % i)
        plt.close()
        
        print 'time', time, 'A', Ax(time)
        print 'el_field', Ax.get_electric_field(time)
        
        time_ar.append(time)
        wf_init = wf_final
        wf_final, dt_new, NK_new = propagate_wave_function(
            wf_init, ham2, NK=NK_new, dt=dt_new, 
            maxel = maxel, regime='TSC', 
            file_out = '../../../../Desktop/pics_2d/TB2_laser/f%03d_2d.png'
            % i)
                
        ix, iy = wf_init.find_average_position()
        xav.append(ix)
        yav.append(iy)
        
        try:
            wfur = make_fourier(wf_final, Nx, Ny)
            plt.pcolor(abs(wfur))
            plt.colorbar()
        except:
            plt.plot(np.sqrt(abs(np.fft.fft(wf_final.wf1d))))
        plt.savefig('../../../../Desktop/pics_2d/TB2_laser/fur_%d.png' % i)
        plt.close()
        
        plt.plot(abs(wf_final.wf1d))
        plt.savefig('../../../../Desktop/pics_2d/TB2_laser/wf1D_%d.png' % i)
        plt.close()
        
        wf.append(abs(wf_final.wf1d))
        
        Ex = Ax.get_electric_field(time)[0]
        
        U.append(Ex * np.linspace(0, Nx * np.sqrt(3) * a, 50))
        
        J = current(wf_final)
        print "current", J
        
        if time > 20 * 10**(-15):
            break
    
    print wf
    plt.imshow(np.array(wf))
    plt.axes().set_aspect('auto')
    plt.show()
    ##plt.savefig('../../../../Desktop/pics_2d/TB2_laser/time_space.png')
    ##plt.close()
    
    U = np.array(U)
    X,Y = np.meshgrid(time_ar, np.linspace(0, Nx * np.sqrt(3) * a, 50))

    plt.subplot(1,2,1)
    plt.plot(time_ar, xav)
    plt.pcolor(X,Y,U.T)
    plt.colorbar()
    
    plt.xlabel(r'$t(s)$')
    plt.ylabel(r'$<x>$')
    plt.xlim(0, max(time_ar))
    plt.ylim(0, Nx * np.sqrt(3) * a)
    
    plt.subplot(1,2,2)
    plt.plot(time_ar, yav)
    plt.xlabel(r'$t(s)$')
    plt.ylabel(r'$<y>$')
    plt.xlim(0, max(time_ar))
    plt.ylim(0, wf_final.coords[Ny][1])
    plt.ylim(0, Ny * 3/4 * a)
    plt.show()    
    return None

propagate_graphene_flatpulse()
