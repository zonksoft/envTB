""" This module contains all functions needed for analyzing 
    of time-dependent problem. It includes functions working
    with output files."""

import numpy as np
import matplotlib.pylab as plt
import envtb.utility.fourier as fourier
import envtb.wannier90.w90hamiltonian as w90
reload(w90)
reload(fourier)
from envtb.utility.fourier import GNRSimpleFourierTransform
import copy
import os

class NumericalData(object):

    def __init__(self, dic_desc, file_name_wf, file_name_eig, file_name_cc, file_name_exp):
        self.dic_desc = dic_desc
        self.file_name_wf = file_name_wf
        self.file_name_eig = file_name_eig
        self.file_name_cc = file_name_cc
        self.file_name_exp = file_name_exp
        try:
            self.eig = self.__get_energy_spectrum(file_name_eig)
        except:
            print 'exception', file_name_eig
        try:
            self.expansion = self.__get_expansion(file_name_exp)
        except:
            print 'exception', file_name_exp
        try:
            self.x, self.y, self.j_x, self.j_y = self.__get_time_coords_current(file_name_cc)
        except:
            print 'exception', file_name_cc
        try:
            self.j_x_f, self.j_y_f = self.__get_fourier_of_current(self.j_x, self.j_y)
        except:
            self.j_x_f = 0.0
            self_j_y_f = 0.0
        try:
            self.wf = self.get_wave_function_from_file(file_name_wf)[1]
        except:
            print 'exception', file_name_wf
        try:
            self.ro_0, self.ro = self.calculate_density_matrix(self.expansion)
        except:
            print 'exception', 'den_matrix'

    def __get_energy_spectrum(self, file_name):
        f_eig = open(file_name, 'r')
        eig = [eval(ln.split('   ')[0]) for ln in f_eig]
        f_eig.close()
        return np.array(eig)

    def __get_expansion(self,file_name):
        f_exp = open(file_name, 'r')
        ln_lines=f_exp.readlines()
        c = [eval(ln) for ln in ln_lines[:-1]]
        f_exp.close()
        return np.array(c)

    def __get_time_coords_current(self, file_name):
        f_coords = open(file_name, 'r')
        x = []
        y = []
        j_x = []
        j_y = []
        for ln in f_coords:
            lnS = ln.split('   ')
            x.append(float(lnS[0]))
            y.append(float(lnS[1]))
            j_x.append(float(lnS[2]))
            j_y.append(float(lnS[3]))
        f_coords.close()
        return x, y, j_x, j_y

    def __get_fourier_of_current(self,j_x, j_y):
        j_x_w = np.fft.fft(np.array(j_x))
        j_y_w = np.fft.fft(np.array(j_y))
        return j_x_w, j_y_w

    @staticmethod
    def get_wave_function_from_file(file_name, Nwf=0):
        fin = open(file_name,'r')
        ln = fin.readlines()[Nwf]
        lnS = ln.split('   ')
        tm = float(lnS[0])
        wf = np.array(eval(lnS[1]))
        fin.close()
        return tm, wf

    def calculate_density_matrix(self, c):
        ro_0 = np.array([[c[0,i]*c[0,j].conjugate() for i in xrange(len(c[0,:]))] for j in xrange(len(c[0,:]))])
        ro = np.array([[c[-2,i]*c[-2,j].conjugate() for i in xrange(len(c[-2,:]))] for j in xrange(len(c[-2,:]))])
        return ro_0, ro

# end class NumericalData


class PlotNumericalData(object):

    def __init__(self, data):
        dt = data.dic_desc['dt']
        self.time = np.arange(0, (len(data.x)-1)*dt, dt)
        self.A_array, self.E_array = self.calculate_pulse_field(self.time, data.dic_desc['A'])
        self.simpleft = None

    def change_time(self, data, dt):
        return self.copy_ins(data, dt)

    def copy_ins(self, data, dt):
        ins = copy.copy(self)
        self.time = np.arange(0, (len(data.x)-1)*dt, dt)
        self.A_array, self.E_array = self.calculate_pulse_field(self.time, data.dic_desc['A'])
        return ins

    def calculate_pulse_field(self, time_array, Ax):
        Ax_array = [Ax(t)[0] for t in time_array]
        Ay_array = [Ax(t)[1] for t in time_array]
        A_array=[Ax_array, Ay_array]
        Ex_array = [10**(-12)*Ax.get_electric_field(t)[0] for t in time_array]
        Ey_array = [10**(-12)*Ax.get_electric_field(t)[1] for t in time_array]
        E_array=[Ex_array, Ey_array]
        return A_array, E_array

    @staticmethod
    def plot_wave_function(wf, Nx, Ny, file_to_save='wf.png',figuresize=(10,10)):
        wf_resized = wf.reshape(Nx, Ny)
        if figuresize:
            plt.figure(figsize=figuresize) # in inches!
        plt.imshow(np.abs(wf_resized.T), interpolation='bilinear', aspect=0.42)
        if file_to_save:
            plt.savefig(file_to_save)
            plt.close()
        return None

    @staticmethod
    def plot_eigen_spectrum(eig, laser_freq_eV=0.0, file_to_save='eig.png', figuresize=(10,10)):
        plt.figure(figsize=figuresize) # in inches!
        plt.plot(eig.real, 'o', ms=2)
        plt.ylabel(r'$E, eV$', fontsize=26)
        plt.axhline(y=laser_freq_eV, color='r', ls='--')
        plt.axhline(y=0, color='r', ls='--')
        if file_to_save:
            plt.savefig(file_to_save)
            plt.close()
        return None

    @staticmethod
    def plot_expansion(time, A_array, E_array, eig, c, file_to_save='exp.png',figuresize=(10,10)):
        from matplotlib.colors import LogNorm
        plt.figure(figsize=figuresize) # in inches!

        plt.subplot(1,2,1)
        plt.plot(A_array[0], time, label = r'$A_x$')
        plt.plot(E_array[0], time, label = r'$E_x$')
        plt.plot(A_array[1], time, label = r'$A_y$', ls = '--')
        plt.plot(E_array[1], time, label = r'$E_y$', ls = '--')
        plt.ylim(0, max(time))
        plt.ylabel(r'$t, s$', fontsize=26)
        plt.legend(prop={'size':16})

        plt.subplot(1,2,2)
        X,Y = np.meshgrid(eig.real, time)
        plt.pcolor(X, Y, c, norm=LogNorm(vmin=1.0e-3, vmax=1.0))
        plt.xlim(min(eig.real), max(eig.real))
        plt.ylim(0.0, max(time))
        plt.colorbar()
        plt.xlabel(r'$E, eV$', fontsize=26)
        plt.xlim(-0.21,0.21)
        if file_to_save:
            plt.savefig(file_to_save)
            plt.close()
        return None

    @staticmethod
    def plot_density_matrix(ro_0, ro, file_to_save='denmat.png', figuresize=(10,10)):
        plt.figure(figsize=figuresize) # in inches!

        plt.subplot(1,2,1)
        plt.imshow(ro_0, interpolation='nearest')
        plt.xlim(0, 300)
        plt.ylim(0, 300)

        plt.subplot(1,2,2)
        plt.imshow(ro, interpolation='nearest')
        plt.xlim(0, 300)
        plt.ylim(0, 300)

        if file_to_save:
            plt.savefig(file_to_save)
            plt.close()
        return None

    @staticmethod
    def plot_coords(time, A_array, E_array, x, y, file_to_save='coords.png', figuresize=(10,10)):
        plt.figure(figsize=figuresize) # in inches!

        plt.subplot(1,3,1)
        plt.plot(A_array[0], time, label = r'$A_x$')
        plt.plot(E_array[0], time, label = r'$E_x$')
        plt.plot(A_array[1], time, label = r'$A_y$', ls = '--')
        plt.plot(E_array[1], time, label = r'$E_y$', ls = '--')
        plt.ylim(0, max(time))
        plt.legend(prop={'size':16})
        plt.ylabel(r'$t, s$', fontsize=26)

        plt.subplot(1,3,2)
        print len(x), len(time)
        try:
            plt.plot(x, time, label=r'x')
        except:
            print len(x), len(time)
            plt.plot(x[:-1], time, label=r'x')
        plt.ylim(0, max(time))
        norm = (max(np.array(x)-x[0]))/max(E_array[0])
        plt.plot(np.array(E_array[0])*norm+x[0], time, label = r'$E_x$', ls='--')
        plt.xlabel(r'$<x>$', fontsize=26)
        plt.legend(prop={'size':16})

        plt.subplot(1,3,3)
        try:
            plt.plot(y, time, label=r'y')
        except:
            plt.plot(y[:-1], time, label=r'y')
        norm = (max(np.array(y)-y[0]))/max(E_array[1])
        plt.plot(np.array(E_array[1])*norm+y[0], time, label = r'$E_y$', ls='--')
        plt.ylim(0, max(time))
        plt.xlabel(r'$<y>$', fontsize=26)
        plt.legend(prop={'size':16})

        if file_to_save:
            plt.savefig(file_to_save)
            plt.close()
        return None

    @staticmethod
    def plot_current(time, A_array, E_array, j_x, j_y, file_to_save='current.png', figuresize=(10,10)):
        plt.figure(figsize=figuresize) # in inches!

        plt.subplot(1,3,1)
        plt.plot(A_array[0], time, label = r'$A_x$')
        plt.plot(E_array[0], time, label = r'$E_x$')
        plt.plot(A_array[1], time, label = r'$A_y$', ls = '--')
        plt.plot(E_array[1], time, label = r'$E_y$', ls = '--')
        plt.ylim(0, max(time))
        plt.legend(prop={'size':16})
        plt.ylabel(r'$t, s$', fontsize=26)

        plt.subplot(1,3,2)
        try:
            plt.plot(j_x, time, label=r'$j_x$')
        except:
            plt.plot(j_x[:-1], time, label=r'$j_x$')
        plt.ylim(0, max(time))
        norm = max(j_x)/max(A_array[0])
        plt.plot(np.array(A_array[0])*norm, time, label = r'$A_x$', ls='--')
        plt.xlabel(r'$<j_x>$', fontsize=26)
        plt.legend(prop={'size':16})
        #plt.xlim(-0.3,0.3)

        plt.subplot(1,3,3)
        try:
            plt.plot(j_y, time, label=r'$j_y$')
        except:
            plt.plot(j_y[:-1], time, label=r'$j_y$')
        plt.ylim(0, max(time))
        norm = max(j_y)/max(A_array[1])
        plt.plot(np.array(A_array[1])*norm, time, label = r'$A_y$', ls='--')
        plt.xlabel(r'$<j_y>$', fontsize=26)
        plt.legend(prop={'size':16})
        #plt.xlim(-0.001,0.001)
        if file_to_save:
            plt.savefig(file_to_save)
            plt.close()
        return None

    @staticmethod
    def plot_current_fourier(time, j_x, j_y, Ax, laser_freq_eV=0.0, file_to_save='curfour.png', figuresize=(10,10)):
        plt.figure(figsize=figuresize) # in inches!

        dt_f = time[1]-time[0]

        j_x_ext = np.zeros(3* len(j_x), dtype=complex)
        j_x_ext[1*len(j_x): 2*len(j_x)] = j_x[:]
        j_x_ext_w = np.fft.fft(j_x_ext)
        freq = np.fft.fftfreq(len(j_x_ext_w), d=dt_f)*4.1357*10**(-15)

        #fig.add_subplot(1,1,1)
        Ax.plot_fourier_transform()

        plt.subplot(1,2,1)
        plt.axvline(x=laser_freq_eV, color='r',ls='--')
        plt.plot(freq[:250], abs(j_x_ext_w)[:250]*10**(-7), label = r'$j_x$')
        plt.xlim(0.0, 0.25*max(freq))
        plt.legend(prop={'size':16})

        plt.subplot(1,2,2)
        plt.legend(prop={'size':16})

        if file_to_save:
            plt.savefig(file_to_save)
            plt.close()

        return None

    @staticmethod
    def plot_wave_function_fourier(wf, plotdata, Nx, Ny, BZ=True, file_to_save='wffour.png', figuresize=(10,10)):

        if figuresize:
            plt.figure(figsize=figuresize) # in inches!
        if not plotdata.simpleft:
            nnfile = '/home/larisa/envtb-data/data/02_graphene_3rdnn/graphene1stnnlist.dat'
            plotdata.simpleft = GNRSimpleFourierTransform(Ny, Nx, nnfile)

        fourier_transform = plotdata.simpleft.fourier_transform(wf)
        four_arr = np.abs(plotdata.simpleft.roll_2d_center(fourier_transform[...,0])).transpose()
        max_f = np.max(four_arr)
        plt.imshow(four_arr, vmax=1.0*max_f, extent=[0, plotdata.simpleft.maxkx, 0, plotdata.simpleft.maxky], 
                   origin='bottom', interpolation='nearest', aspect=1.0)
        if BZ:
            BZ = plotdata.simpleft.get_brillouin_zone()
            plt.plot(BZ[0], BZ[1], 'w', ls='--')
        #print BZ[0]-simpleft.maxkx/2
        plt.xlim(36.0,41.0)
        plt.ylim(20,24)
        plt.xlabel(r'$k_x$', fontsize=26)
        plt.ylabel(r'$k_y$', fontsize=26)

        if file_to_save:
            plt.savefig(file_to_save)
            plt.close()
        return None

    @staticmethod
    def plot_wave_function_with_fourier(wf, plotdata, Nx, Ny, file_to_save='wfandfour.png', figuresize=(20,10)):
        if figuresize:
            plt.figure(figsize=figuresize) # in inches!

        plt.subplot(1,2,1)
        PlotNumericalData.plot_wave_function(wf, Nx, Ny, file_to_save=None, figuresize=None)

        plt.subplot(1,2,2)
        PlotNumericalData.plot_wave_function_fourier(wf, plotdata, Nx, Ny, BZ=True, file_to_save=None, figuresize=None)

        if file_to_save:
            plt.savefig(file_to_save)
            plt.close()
        return None

    @staticmethod
    def plot_wave_functions_stack(file_name, Nx, Ny, nx_s=3, ny_s=10, time_step=2, file_to_save='wf_stack.png', figuresize=(20,30)):

        plt.figure(figsize=figuresize)
        for i in xrange(nx_s * ny_s):
            plt.subplot(ny_s,nx_s,i+1)
            tm, wf = NumericalData.get_wave_function_from_file(file_name, Nwf=i*time_step)
            PlotNumericalData.plot_wave_function(wf, Nx, Ny, file_to_save=None, figuresize=None)
            plt.title('%.2E' % tm)

        if file_to_save:
            plt.savefig(file_to_save)
            plt.close()

        return None

#end class PlotNumericalData


def make_header(dic_desc, figs=[], file_name='header.tex'):
    latexfile = open(file_name, 'w')
    shape = dic_desc['shape']; amp = dic_desc['laser_amp']
    freq = dic_desc['laser_freq']*10**(-12); freq_eV = dic_desc['laser_freq_eV']
    CEP = dic_desc['CEP']; Nc = dic_desc['Nc']
    direction_x = dic_desc['direction'][0]; direction_y = dic_desc['direction'][1]
    dt = dic_desc['dt']; Nx = dic_desc['Nx']; Ny = dic_desc['Ny']; ID = dic_desc['ID']
    latexfile.write(r"""
%% Automatically generated LaTeX file
\documentclass[a4paper,12pt]{article}

\usepackage[english]{babel}
\usepackage{graphicx}

\begin{document}
Description of the system\\
\\
Pulse:\\
shape: %(shape)s\\
amplitude (V/m): %(amp).2E\\
frequency (THz): %(freq).2E\\
frequency in eV: %(freq_eV).5f\\
CEP: %(CEP).f\\
direction: [%(direction_x).f, %(direction_y).f ] \\
Nc: %(Nc)d\\
dt: %(dt).2E\\
\\
System (ZGNR):\\
Nx: %(Nx)d\\
Ny: %(Ny)d\\
eigenstate id: %(ID)d\\
""" % vars())

    latexfile.close()
    for fig_file in figs:
        add_figure(file_name=file_name, file_with_figure=fig_file)

    latexfile = open(file_name, 'a')
    latexfile.write(r"""
\end{document}
""")
    latexfile.close()


def add_figure(file_name, file_with_figure):
    latexfile = open(file_name, 'a')
    latexfile.write(r"""
    \begin{figure}[h]
        \vspace{-50pt}
        \centering
        \advance\leftskip-3cm
        \includegraphics[width=1.0\paperwidth]{%(file_with_figure)s}
    \end{figure}
    """%vars())
    latexfile.close()
    return None

def make_pdf(dic_desc, directory='\tmp\tmp', file_out='test.tex'):
    #fast_plot(numdata, plotdata, save=True)
    import glob
    pngfiles = sorted(glob.glob("*.png"))
    make_header(dic_desc, figs=pngfiles, file_name=file_out)
    import subprocess
    #subprocess.call('convert -quality 100 `ls *.png` %(file_out)s' % vars(), shell=True)

    if not os.path.isdir(directory):
        os.mkdir(directory)
    path, folder = os.path.split(directory)

    subprocess.call('pdflatex %(file_out)s' % vars(), shell=True)
    subprocess.call('mv *.png  *.tex %(directory)s' % vars(), shell=True)
    subprocess.call('mv *.pdf %(path)s' % vars(), shell=True)
    return None

def fast_plot(numdata, plotdata, save=False):
    if not save:
         #plotdata.plot_wave_function(wf=numdata.wf,
        #         Nx=numdata.dic_desc['Nx'], Ny=numdata.dic_desc['Ny'],
        #            file_to_save=None, figuresize=(8,10))
         plotdata.plot_eigen_spectrum(eig=numdata.eig,
                    laser_freq_eV=numdata.dic_desc['laser_freq_eV'],
                    file_to_save=None, figuresize=(10,10))
         plotdata.plot_expansion(time=plotdata.time, A_array=plotdata.A_array,
                    E_array=plotdata.E_array, eig=numdata.eig,
                    c=numdata.expansion,
                    file_to_save=None, figuresize=(20,10))
         plotdata.plot_density_matrix(ro_0=numdata.ro_0, ro=numdata.ro,
                    file_to_save=None, figuresize=(20,10))
         plotdata.plot_coords(time=plotdata.time, A_array=plotdata.A_array,
                    E_array=plotdata.E_array, x=numdata.x, y=numdata.y,
                    file_to_save=None, figuresize=(20,10))
         plotdata.plot_current(time=plotdata.time, A_array=plotdata.A_array,
                    E_array=plotdata.E_array, j_x=numdata.j_x, j_y=numdata.j_y,
                    file_to_save=None, figuresize=(20,10))
         plotdata.plot_current_fourier(time=plotdata.time,
                    j_x=numdata.j_x, j_y=numdata.j_y, Ax=numdata.dic_desc['A'],
                    laser_freq_eV=numdata.dic_desc['laser_freq_eV'],
                    file_to_save=None, figuresize=(20,10))
         plotdata.plot_wave_functions_stack(file_name=numdata.file_name_wf,
                    Nx=numdata.dic_desc['Nx'], Ny=numdata.dic_desc['Ny'],
                    nx_s=3, ny_s=10, time_step=2,
                    file_to_save=None, figuresize=(25,35))
    else:
        #plotdata.plot_wave_function(wf=numdata.wf,
        #         Nx=numdata.dic_desc['Nx'], Ny=numdata.dic_desc['Ny'],
        #            file_to_save='1_wf.png', figuresize=(8,10))
        try:
            plotdata.plot_eigen_spectrum(eig=numdata.eig,
                    laser_freq_eV=numdata.dic_desc['laser_freq_eV'],
                    file_to_save='2_eig.png', figuresize=(10,10))
        except:
            print 'Failed to plot!'
            pass
        try:
            plotdata.plot_expansion(time=plotdata.time, A_array=plotdata.A_array,
                    E_array=plotdata.E_array, eig=numdata.eig,
                    c=numdata.expansion,
                    file_to_save='3_exp.png', figuresize=(20,10))
        except:
            print 'Failed to plot!'
            pass
        try:
            plotdata.plot_density_matrix(ro_0=numdata.ro_0, ro=numdata.ro,
                    file_to_save='4_den.png', figuresize=(20,10))
        except:
            print 'Failed to plot!'
            pass
        try:
            plotdata.plot_coords(time=plotdata.time, A_array=plotdata.A_array,
                    E_array=plotdata.E_array, x=numdata.x, y=numdata.y,
                    file_to_save='5_coords.png', figuresize=(20,10))
        except:
            print 'Failed to plot!'
            pass
        try:
            plotdata.plot_current(time=plotdata.time, A_array=plotdata.A_array,
                    E_array=plotdata.E_array, j_x=numdata.j_x, j_y=numdata.j_y,
                    file_to_save='6_cur.png', figuresize=(20,10))
        except:
            print 'Failed to plot!'
            pass
        try:
            plotdata.plot_current_fourier(time=plotdata.time,
                    j_x=numdata.j_x, j_y=numdata.j_y, Ax=numdata.dic_desc['A'],
                    laser_freq_eV=numdata.dic_desc['laser_freq_eV'],
                    file_to_save='7_cur_f.png', figuresize=(20,10))
        except:
            print 'Failed to plot!'
            pass
        try:
            plotdata.plot_wave_functions_stack(file_name=numdata.file_name_wf,
                    Nx=numdata.dic_desc['Nx'], Ny=numdata.dic_desc['Ny'],
                    nx_s=3, ny_s=10, time_step=2,
                    file_to_save='1_wf_stack.png', figuresize=(25,35))
        except:
            print 'Failed to plot!'
            pass
    return None