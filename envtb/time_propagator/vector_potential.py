import numpy as np
import matplotlib.pylab as plt

light_speed = 3 * 10**(8)
print 'Agrh!'

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
        tarr = np.linspace(-0.1 / self.frequency, 7 * np.pi / self.frequency, 100)
        plt.subplot(1,2,1)
        Ex = [self.get_electric_field(t)[0] for t in tarr]
        plt.plot(tarr, Ex, label = r'$E_x$')
        plt.subplot(1,2,2)
        Ey = [self.get_electric_field(t)[1] for t in tarr]
        plt.plot(tarr, Ey, label = r'$E_y$')
    
    def plot_pulse(self):
        tarr = np.linspace(-0.1 / self.frequency, 7 * np.pi / self.frequency, 100)
        plt.subplot(1,2,1)
        Ax = [self.__call__(t)[0] for t in tarr]
        plt.plot(tarr, Ax, label=r'$A_x$')
        plt.xlabel(r'$t, s$')
        plt.subplot(1,2,2)
        Ay = [self.__call__(t)[1] for t in tarr]
        plt.plot(tarr, Ay, label=r'$A_y$')
        plt.xlabel(r'$t, s$')
    
    def __call__(self, t):
        pass
    
class VectorPotentialWave(VectorPotential):
    
    def __init__(self, amplitude_E0, frequency, direction=[1.0,0.0]):
        
        """
        amplitude_E0: is a maximum field strength
        """
        self.amplitude= amplitude_E0
        self.frequency = frequency
        norm = np.sqrt(direction[0]**2+direction[1]**2)
        self.direction = np.array(direction)/norm
        
    def __call__(self, t):
        VecPot = self.amplitude * light_speed / self.frequency * np.sin(self.frequency * t)
        return [VecPot * self.direction[0], VecPot * self.direction[1]]
       
    
class LP_FlatTopPulse(VectorPotential):
    
    def __init__(self, amplitude_E0, frequency, direction=[1.0,0.0]):
        
        """
        amplitude_E0: is the maximum field strength
        """
        self.amplitude= amplitude_E0
        self.frequency = frequency
        norm = np.sqrt(direction[0]**2+direction[1]**2)
        self.direction = np.array(direction)/norm
            
    def __call__(self, t):
        
        if t < 0:
            return [0, 0]
        elif t >= 0 and t < np.pi/self.frequency:
            VecPot = self.amplitude * light_speed * t / np.pi * np.sin(self.frequency * t)
            return [VecPot * self.direction[0], VecPot * self.direction[1]]
        else:
            VecPot = self.amplitude * light_speed / self.frequency * np.sin(self.frequency * t)
            return [VecPot * self.direction[0], VecPot * self.direction[1]]

    
class LP_SinSqEnvelopePulse(VectorPotential):
    
    def __init__(self, amplitude_E0, frequency, Nc=1, cep=0.0, direction=[1.0,0.0]): 
        
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
        norm = np.sqrt(direction[0]**2 + direction[1]**2)
        self.direction = np.array(direction) / norm
        self.CEP = cep
    
    def __call__(self, t):
        
        if t >= 0 and t < self.pulse_duration:
            VecPot = -self.amplitude * light_speed / self.frequency *\
                    np.sin(self.frequency * t / 2./self.Nc)**2 *\
                    np.sin(self.frequency * t + self.CEP)
            return [VecPot * self.direction[0], VecPot * self.direction[1]]
        else:
            return [0.,0]
        
    def plot_envelope(self):
        tarr = np.linspace(0., self.pulse_duration, 100)
        plt.subplot(1,2,1)
        A = [self.amplitude * light_speed / self.frequency *\
                np.sin(self.frequency * t / 2./self.Nc)**2 * self.direction[0] for t in tarr]
        plt.plot(tarr, A, '--')
        plt.subplot(1,2,2)
        A = [self.amplitude * light_speed / self.frequency *\
                np.sin(self.frequency * t / 2./self.Nc)**2 * self.direction[1] for t in tarr]
        plt.plot(tarr, A, '--')
        
                   