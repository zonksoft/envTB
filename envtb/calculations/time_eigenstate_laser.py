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
dt = 0.004 * 10**(-12)
NK = 12
laser_freq = 10**(12)
laser_amp = 1.0 * 10**(-2)
Nc = 2 #number of laser cycles

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

def propagate_graphene_pulse(Nx=30, Ny=30, frame_num=1500):
    """
    Since in lanczos in the exponent exp(E*t/hbar) we are using E in eV    
   
    """
    ham = envtb.ldos.hamiltonian.HamiltonianGraphene(Nx,Ny)
    w, v = ham.eigenvalue_problem()
    isort = np.argsort(w)
    
    wsort = np.sort(w)
    vsort = np.zeros(v.shape, dtype=complex)
    for i in xrange(len(isort)):
        vsort[:,i] = v[:,isort[i]]
        
    plt.subplot(1,2,1)
    plt.plot(w.real, 'o', ms=2)
    plt.subplot(1,2,2)
    plt.plot(wsort.real, 'o', ms=2)
    plt.show()
    
    Nstate = 545
    
    envtb.ldos.plotter.Plotter().plot_density(
        vector=abs(v[:, Nstate]), coords=ham.coords)
    plt.show()
    
    # Make vector potential
    Ax = envtb.time_propagator.vector_potential.LP_SinSqEnvelopePulse(
        amplitude_E0 = laser_amp, frequency = laser_freq, Nc = Nc)
    Ax.plot_pulse()
    Ax.plot_envelope()
    Ax.plot_electric_field()
    plt.show()
    
    #main loop
    wf_out = open('wave_functions.out','w')
    dt_new = dt
    NK_new = NK
    time = 0.0
    
    wf_final = envtb.time_propagator.wave_function.WaveFunction(vec=v[:, Nstate],coords=ham.coords)
    maxel = max(wf_final.wf1d)
    wf_out.writelines(`time`+'   '+`wf_final.wf1d.tolist()`+'\n')
    
    
    for i in xrange(3500):
        print 'frame %(i)d' % vars()
        time += dt_new
        ham2 = ham.apply_vector_potential(Ax(time))
        print 'time', time, 'A', Ax(time)
        wf_init = wf_final
        wf_final, dt_new, NK_new = propagate_wave_function(
            wf_init, ham2, NK=NK_new, dt=dt_new, 
            maxel = maxel, regime='TSC', 
            file_out = directory+'f%03d_2d.png' % i)
        
        #make expansion
        if np.mod(i,10) == 0:
            
            #tar.append(time)
            wf_out.writelines(`time`+'   '+`wf_final.wf1d.tolist()`+'\n')
           
    wf_out.close()
    

def data_analysis(file_name='/home/larisa/progs/envTB/envtb/calculations/wave_functions.out', Nx=30, Ny=30, Nstate=545):
    from matplotlib.colors import LogNorm
    ham = envtb.ldos.hamiltonian.HamiltonianGraphene(Nx,Ny)
    w, v = ham.eigenvalue_problem()
    isort = np.argsort(w)
    
    wsort = np.sort(w)
    vsort = np.zeros(v.shape, dtype=complex)
    for i in xrange(len(isort)):
        vsort[:,i] = v[:,isort[i]]
    
    fin = open(file_name,'r')
    tar = []
    wf_arr = []
    anarr = []
    for ln in fin:
        tar.append(float(ln.split('   ')[0]))
        wf_arr.append(np.array(eval(ln.split('   ')[1])))
    
    #expansion
    for j in xrange(len(wf_arr)):
        anarr.append([np.abs(np.dot(np.conjugate(np.transpose(vsort[:,i])), wf_arr[j])) for i in xrange(len(w))])
    
    Ax = envtb.time_propagator.vector_potential.LP_SinSqEnvelopePulse(
        amplitude_E0 = laser_amp, frequency = laser_freq, Nc = Nc)
    plt.subplot(1,2,1)
    Ax.plot_pulse()
    plt.ylim(0, max(tar))
    
    plt.subplot(1,2,2)
    X,Y = np.meshgrid(wsort, tar)
    plt.pcolor(X, Y, anarr, norm=LogNorm(vmin=1.0e-3, vmax=1.0))
    plt.xlim(min(wsort), max(wsort))
    plt.ylim(0.0, max(tar))
    plt.colorbar()
    plt.show()
    
    
        
data_analysis()
#propagate_graphene_pulse()
