import numpy as np
import matplotlib.pylab as plt

light_speed = 3 * 10**(8)

class VectorPotential:
    
    def __init__(self):
        self.frequency = None
        self.amplitude = None
        self.pulse_duration = None
            
    def get_electric_field(self, t):
        dt = 0.01 * 10**(-16)
        return [-(self(t + dt)[0] - self(1.*t)[0]) / dt / light_speed, -(self(t + dt)[1] - self(1.*t)[1]) / dt / light_speed]
    
    def get_magnetic_filed(self):
        pass
    
    def plot_electric_field(self):
        tarr = np.linspace(-0.1 / self.frequency, 6 * np.pi / self.frequency, 100)
        plt.subplot(1,2,1)
        Ex = [self.get_electric_field(t)[0] for t in tarr]
        plt.plot(tarr, Ex)
        plt.subplot(1,2,2)
        Ey = [self.get_electric_field(t)[1] for t in tarr]
        plt.plot(tarr, Ey)
    
    def plot_pulse(self):
        tarr = np.linspace(-0.1 / self.frequency, 6 * np.pi / self.frequency, 100)
        plt.subplot(1,2,1)
        Ax = [self.__call__(t)[0] for t in tarr]
        plt.plot(tarr, Ax)
        plt.subplot(1,2,2)
        Ay = [self.__call__(t)[1] for t in tarr]
        plt.plot(tarr, Ay)
    
    def __call__(self, t):
        pass
    
class VectorPotentialWave(VectorPotential):
    
    def __init__(self, amplitude_E0, frequency):
        
        """
        amplitude_E0: is the peak filed strength
        """
        
        self.amplitude= amplitude_E0
        self.frequency = frequency
        
    def __call__(self, t):
        
        return [self.amplitude * light_speed / self.frequency * np.sin(self.frequency * t), 0]
       
    
class LP_FlatTopPulse(VectorPotential):
    
    def __init__(self, amplitude_E0, frequency):
        
        """
        amplitude_E0: is the peak filed strength
        """
        
        self.amplitude= amplitude_E0
        self.frequency = frequency
            
    def __call__(self, t):
        
        if t < 0:
            return [0, 0]
        elif t >= 0 and t < np.pi/self.frequency:
            return [self.amplitude * light_speed * t / np.pi * np.sin(self.frequency * t), 0]
        else:
            return [self.amplitude * light_speed / self.frequency * np.sin(self.frequency * t), 0]  

    
class LP_SinSqEnvelopePulse(VectorPotential):
    
    def __init__(self, amplitude_E0, frequency, Nc = 1): 
        """
        This class contains linearly polarized light filed described
        by the vector potential with a sin^2 envelope 
        
        amplitude_E0: is the peak filed strength
        
        frequency: pulse frequency
        
        Nc: number of cycles
        """
        
        self.amplitude= amplitude_E0
        self.frequency = frequency
        self.Nc = Nc
        self.pulse_duration = 2. * self.Nc * np.pi / self.frequency
        
    
    def __call__(self, t):
        
        if t >= 0 and t < self.pulse_duration:
            return [-self.amplitude * light_speed / self.frequency *\
                    np.sin(self.frequency * t / 2./self.Nc)**2 *\
                    np.sin(self.frequency * t), 0]
        else:
            return [0.,0]
        
    def plot_envelope(self):
        tarr = np.linspace(0., self.pulse_duration, 100)
        plt.subplot(1,2,1)
        A = [self.amplitude * light_speed / self.frequency *\
                np.sin(self.frequency * t / 2./self.Nc)**2 for t in tarr]
        plt.plot(tarr, A, '--')
        plt.subplot(1,2,2)
        A = [0 for t in tarr]
        plt.plot(tarr, A, '--')
        
                   