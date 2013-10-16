import envtb.ldos.hamiltonian
import numpy as np
import envtb.time_propagator.lanczos
import envtb.time_propagator.wave_function
import envtb.time_propagator.vector_potential
import envtb.wannier90.w90hamiltonian as w90hamiltonian

class Propagator(object):
    """docstring for Propagator"""
    def __init__(self, num_error=10**(-18), regime='SIL'):
        self.num_error=num_error
        self.regime=regime
	
    @staticmethod
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

