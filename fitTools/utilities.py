'''
This is a backend module for fitting complex resonator data. 
If you're trying to fit a resonator, best to use the class Resonator defined in resonator.py by 
from fitTools.resonator import Resonator.
'''
import warnings
import numpy as np
import matplotlib.pyplot as plt

def Watt2dBm(x):
    '''
    converts from units of Watts to dBm
    '''
    return 10.*np.log10(x*1000.)

def Volt2dBm(x):
    '''
    converts from rms voltage to dBm, assuming 50 Ohm.
    '''
    return 10*np.log10(x**2*1000/50.)

def VNA2Volt(z):
    '''
    converts from the VNA's complex data to V.
    '''
    return np.abs(z)*np.sqrt(50/1000.)

def VNA2Watt(z):
    '''
    converts from VNA complex data to W.
    '''
    return (np.abs(z)**2)/1000.

def VNA2dBm(z):
    '''
    converts from VNA complex data to dBm
    '''
    return 20*np.log10(np.abs(z))

def dBm2Watt(x):
    '''
    converts from units of dBm to Watts
    '''
    return 10**(x/10.) /1000.	


def dBm2Volt(x):
    '''
    converts from units of dBm to Volts, assuming 50 ohm
    '''
    return np.sqrt(50 * 10**(x/10) / 1000)

class plotting(object):
    '''
    some helper functions for plotting
    '''
    def show(self,savefile=None):
        real = self.z_data_raw.real
        imag = self.z_data_raw.imag
        real_sim = self.z_data_sim.real
        imag_sim = self.z_data_sim.imag
        real_cor = self.z_data.real
        imag_cor = self.z_data.imag
        real_cor_sim = self.z_data_sim_norm.real
        imag_cor_sim = self.z_data_sim_norm.imag
        
        fig, axs = plt.subplots(2, 3, sharex='col', constrained_layout=True,figsize=[10,10])
        axs[0,0].plot(real,imag,'.b',label='rawdata')
        axs[0,0].plot(real_sim,imag_sim,'-r',label='fit')
        axs[0,0].set_title('Raw in complex plane')
        axs[0,0].set_ylabel('Imaginary')
        axs[0,0].set_xlabel('Real')
        axs[0,0].set_aspect('equal')
        axs[0,0].grid()
        axs[1,0].plot(real_cor,imag_cor,'.b')
        axs[1,0].plot(real_cor_sim,imag_cor_sim,'-r')
        axs[1,0].set_title('Normalized data in complex plane')
        axs[1,0].set_ylabel('Imaginary')
        axs[1,0].set_xlabel('Real')
        axs[1,0].set_aspect('equal')
        axs[1,0].grid()
        axs[0,1].plot(self.f_data*1e-9,np.angle(self.z_data_raw),'.b')
        axs[0,1].plot(self.f_data*1e-9,np.angle(self.z_data_sim),'-r')
        axs[0,1].set_title('Raw phase')
        axs[0,1].set_ylabel('Phase [radians]')
        axs[0,1].set_xlabel('Frequency [GHz]')
        axs[0,1].grid()
        axs[1,1].plot(self.f_data*1e-9,np.angle(self.z_data),'.b')
        axs[1,1].plot(self.f_data*1e-9,np.angle(self.z_data_sim_norm),'-r')
        axs[1,1].set_title('Normalized phase')
        axs[1,1].set_ylabel('Phase [radians]')
        axs[1,1].set_xlabel('Frequency [GHz]')
        axs[1,1].grid()
        axs[0,2].plot(self.f_data*1e-9,VNA2dBm(self.z_data_raw),'.b')
        axs[0,2].plot(self.f_data*1e-9,VNA2dBm(self.z_data_sim),'-r')
        axs[0,2].set_title('Log Magnitude response')
        axs[0,2].set_ylabel('Magnitude [dBm]')
        axs[0,2].set_xlabel('Frequency [GHz]')
        axs[0,2].grid()
        axs[1,2].plot(self.f_data*1e-9,1000*VNA2Watt(self.z_data_raw),'.b')
        axs[1,2].plot(self.f_data*1e-9,1000*VNA2Watt(self.z_data_sim),'-r')
        axs[1,2].set_title('Linear Magnitude response')
        axs[1,2].set_ylabel('Magnitude [mW]')
        axs[1,2].set_xlabel('Frequency [GHz]')
        axs[1,2].grid()
        fig.suptitle('Resonator Fitting')
        fig.legend()
        if type(savefile) is str:
            plt.savefig(savefile)
        plt.show()
        
#        plt.subplot(221)
#        plt.plot(real,imag,label='rawdata')
#        plt.plot(real2,imag2,label='fit')
#        plt.xlabel('Re(S21)')
#        plt.ylabel('Im(S21)')
#        plt.legend()
#        plt.subplot(222)
#        plt.plot(self.f_data*1e-9,np.absolute(self.z_data_raw),label='rawdata')
#        plt.plot(self.f_data*1e-9,np.absolute(self.z_data_sim),label='fit')
#        plt.xlabel('f (GHz)')
#        plt.ylabel('|S21|')
#        plt.legend()
#        plt.subplot(223)
#        plt.plot(self.f_data*1e-9,np.angle(self.z_data_raw),label='rawdata')
#        plt.plot(self.f_data*1e-9,np.angle(self.z_data_sim),label='fit')
#        plt.xlabel('f (GHz)')
#        plt.ylabel('arg(|S21|)')
#        plt.legend()
#        plt.show()
        
    def plotcalibrateddata(self):
        real = self.z_data.real
        imag = self.z_data.imag
        plt.subplot(221)
        plt.plot(real,imag,label='rawdata')
        plt.xlabel('Re(S21)')
        plt.ylabel('Im(S21)')
        plt.legend()
        plt.subplot(222)
        plt.plot(self.f_data*1e-9,np.absolute(self.z_data),label='rawdata')
        plt.xlabel('f (GHz)')
        plt.ylabel('|S21|')
        plt.legend()
        plt.subplot(223)
        plt.plot(self.f_data*1e-9,np.angle(self.z_data),label='rawdata')
        plt.xlabel('f (GHz)')
        plt.ylabel('arg(|S21|)')
        plt.legend()
        plt.show()
        
    def plotrawdata(self):
        real = self.z_data_raw.real
        imag = self.z_data_raw.imag
        plt.subplot(221)
        plt.plot(real,imag,label='rawdata')
        plt.xlabel('Re(S21)')
        plt.ylabel('Im(S21)')
        plt.legend()
        plt.subplot(222)
        plt.plot(self.f_data*1e-9,np.absolute(self.z_data_raw),label='rawdata')
        plt.xlabel('f (GHz)')
        plt.ylabel('|S21|')
        plt.legend()
        plt.subplot(223)
        plt.plot(self.f_data*1e-9,np.angle(self.z_data_raw),label='rawdata')
        plt.xlabel('f (GHz)')
        plt.ylabel('arg(|S21|)')
        plt.legend()
        plt.show()

class save_load(object):
    '''
    procedures for loading and saving data used by other classes
    '''
    def _ConvToCompl(self,x,y,dtype):
        '''
        dtype = 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg', 'linmagphasedeg'
        '''
        if dtype=='realimag':
            return x+1j*y
        elif dtype=='linmagphaserad':
            return x*np.exp(1j*y)
        elif dtype=='dBmagphaserad':
            return 10**(x/20.)*np.exp(1j*y)
        elif dtype=='linmagphasedeg':
            return x*np.exp(1j*y/180.*np.pi)
        elif dtype=='dBmagphasedeg':
            return 10**(x/20.)*np.exp(1j*y/180.*np.pi)	 
        else: warnings.warn("Undefined input type! Use 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg' or 'linmagphasedeg'.", SyntaxWarning)

    def add_data(self,f_data,z_data):
        self.f_data = np.array(f_data)
        self.z_data_raw = np.array(z_data)
        
    def cut_data(self,f1,f2):
        def findpos(f_data,val):
            pos = 0
            for i in range(len(f_data)):
                if f_data[i]<val: pos=i
            return pos
        pos1 = findpos(self.f_data,f1)
        pos2 = findpos(self.f_data,f2)
        self.f_data = self.f_data[pos1:pos2]
        self.z_data_raw = self.z_data_raw[pos1:pos2]
        
    def add_fromtxt(self,fname,dtype,header_rows,usecols=(0,1,2),fdata_unit=1.,delimiter=None):
        '''
        dtype = 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg', 'linmagphasedeg'
        '''
        data = np.loadtxt(fname,usecols=usecols,skiprows=header_rows,delimiter=delimiter)
        self.f_data = data[:,0]*fdata_unit
        self.z_data_raw = self._ConvToCompl(data[:,1],data[:,2],dtype=dtype)
        
    def add_fromhdf():
        pass
    
    def add_froms2p(self,fname,y1_col,y2_col,dtype,fdata_unit=1.,delimiter=None):
        '''
        dtype = 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg', 'linmagphasedeg'
        '''
        if dtype == 'dBmagphasedeg' or dtype == 'linmagphasedeg':
            phase_conversion = 1./180.*np.pi
        else: 
            phase_conversion = 1.
        f = open(fname)
        lines = f.readlines()
        f.close()
        z_data_raw = []
        f_data = []
        if dtype=='realimag':
            for line in lines:
                if ((line!="\n") and (line[0]!="#") and (line[0]!="!")) :
                    lineinfo = line.split(delimiter)
                    f_data.append(float(lineinfo[0])*fdata_unit)
                    z_data_raw.append(np.complex(float(lineinfo[y1_col]),float(lineinfo[y2_col])))
        elif dtype=='linmagphaserad' or dtype=='linmagphasedeg':
            for line in lines:
                if ((line!="\n") and (line[0]!="#") and (line[0]!="!") and (line[0]!="M") and (line[0]!="P")):
                    lineinfo = line.split(delimiter)
                    f_data.append(float(lineinfo[0])*fdata_unit)
                    z_data_raw.append(float(lineinfo[y1_col])*np.exp( np.complex(0.,phase_conversion*float(lineinfo[y2_col]))))
        elif dtype=='dBmagphaserad' or dtype=='dBmagphasedeg':
            for line in lines:
                if ((line!="\n") and (line[0]!="#") and (line[0]!="!") and (line[0]!="M") and (line[0]!="P")):
                    lineinfo = line.split(delimiter)
                    f_data.append(float(lineinfo[0])*fdata_unit)
                    linamp = 10**(float(lineinfo[y1_col])/20.)
                    z_data_raw.append(linamp*np.exp( np.complex(0.,phase_conversion*float(lineinfo[y2_col]))))
        else:
            warnings.warn("Undefined input type! Use 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg' or 'linmagphasedeg'.", SyntaxWarning)
        self.f_data = np.array(f_data)
        self.z_data_raw = np.array(z_data_raw)
        
    def save_fitresults(self,fname):
        pass



