# -*- coding: utf-8 -*-
"""A resonator object.

A resonator object which, provided a trace, will fit to a lorentzian
functional form and extract the following parameters:
*location (center frequency)
*amplitude
*gamma (the half width- half max)
*DC offset and slope of background
*Signal to Noise Ratio
*internal, external, and total quality factor
Resonators can be printed and sorted. Use the search function
to return a list of all resonators in a trace. There are functions
to return a trace with the background and/or noise removed.

ToDo
----
* The initial guess for the center frequency should be made at the point
  where the derivative of a smoothed function is the greatest, not where the
  absolute value of transmission reaches an extremmum. [check]
* Need to find true min when guess is made

ChangeLog
---------
1.1.0:
    * Added the option to choose not to rotate complex data. This is good for 
      when the data was acquired after dividing a memory trace of the background.
    * updated the way rotate finds the optimal rotation angle
    * increased the savgol filter window from 4% of trace length to 5%
    * fixed a bug that sent an empty data set to fit when the guess for FWHM fails.
    * added functionality to calculate internal and external Q, based on 
      normalized amplitude, amp/dc offset
1.0.9:
    * Change initial guess for FWHM in fit_data to work for both L2 and L4 
      resonators
    * Change initial guess for peak_index to just take np.argmin or np.argmax
    * Added rotate function. If Resonator is called with magnitude = False,
      automatically rotates complex data such that the real part is Lorentzian,
      then feeds the real part of the data to make_guess and fit_data.
1.0.8:
    * Change tick label size on X axis to 12 from 14. Labels were
      overlapping
    * Fix bug in fit_lorentzian where DC offset was calculated from start
      of frequency trace.  Caused fit parameter to be off when you change
      window size of fit
    * Normalize fitted data so FWHM is properly found
1.0.7:
    * Add tqdm progress bars for the long-running data cleaning
      and fitting steps
1.0.6:
    * Now checks complex data agains np.dtype('complex128') which is
      how Labber now seems to save complex data. (used to be 64 bit)
1.0.5:
    * Add ringdown time as a property
1.0.4:
    * Search function now cannot return a double hit on the same resonator.
      It used to do this if some small feature was left after subtracting
      a previously found resonator from the trace. [JIB]
    * show() now looks prettier
1.0.3:
    * Bugfix in fitting to correct DC offsets. [JIB]
1.0.2:
    * Bugfix & improvements on how initial guess is made. [JIB]
    * Now fitting lorentzian over small window of trace. Gives better
      convergence for wideband data.[JIB]
1.0.1:
    * Better conditioning of input data from PNA. [JIB]

Created on Wed Jan 17 13:50:11 2018
Last update: 05/09/2018
"""

__author__ = 'James Basham, Jeff Grover, James Farmer'
__version__ = '1.0.9'

import gc
from time import time
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from tqdm import tqdm
#import Labber

#from acquisition import LogFile


class Resonator:

    class Kind(Enum):
        L2 = 1  # A Lambda/2 resonator
        L4 = 2  # A Lambda/4 resonator

    class FitError(BaseException):
        pass

    def __init__(self, x=[], y=[], magnitude = False, rotate = True, 
                 kind='L4', force_fit=False, fit_all=True):
        """Construct a Resonator object

        Parameters:
        ----------
        x: numpy.array
            frequency data
        y: numpy.array
            S21 data
        magnitude: False to fit the real part of complex data, True to fit mag
        rotate: True to attempt rotation of real part to lorentzian,
            False to leave real part as is (good when you used data/memory)
        frequency : float, optional
            Estimated resonant frequency
        kind: str, optional
            Type of resonator.  Enter 'L2' for Lambda/2 or 'L4'
            for a Lambda/4
        force_fit : bool
            accept fit even if it isn't good
        fit_all : bool
            Fit the whole trace.  If False, only to fit peak +- FWHM

        Example:
        --------
        filepath = ('C:\\git\\qeo_measurement_data\\Leiden5\\Cygnus_10\\'
            '2018\\05\\Data_0522\\Lyanna-Cygnus_10-1527012322.hdf5')
        f = Labber.LogFile(filepath)
        print('File:   ', filepath.split('\\')[-1])
        print('Comment:', f.getComment())
        print('='*40)
        freq, S21 = f.getTraceXY()
        del(f)
        res = Resonator.Resonator(freq, S21)
        print(res)
        """
        self.fit_all = fit_all
        self.x = np.array(x).flatten()
        self.y = np.array(y).flatten()
        # Suggest usage of Enum class but allow string of valid supported type
        if kind == self.Kind.L2:
            kind = 'L2'
        elif kind == self.Kind.L4:
            kind = 'L4'
        kind = str.upper(kind)
        if (kind not in ['L2', 'L4']):
            print('"' + (str(kind) + '" is not a supported resonator kind.' +
                  'Lambda/4 was chosen instead.'))
            kind = 'L4'
        self.kind = kind
        # Fitted values should have values of 0 or -1 to indicate
        # a fit has not yet been attempted
        self.location = np.nan
        self.fwhm = np.nan
        self.amplitude = np.nan
        self.Q = np.nan
        self.snr = np.nan
        self.fit_values = {}
        self.error = {}
        self.fit_found = False

        if not (self.x.size or self.y.size):
            # If no trace was provided, set a standard width and skip fitting.
            self.gamma = 1e6
            return None
        # If a trace was provided, try to fit it
        # convert to magnitude if needed
        if magnitude:
            self.y = np.abs(y)
        percentWindow = [2,3,4,5,6,7,8]
        for p in percentWindow:
            if rotate and not magnitude:
#                print('attempting rotation')
                self.make_guess(filter_percent=p)
                self.rotate()
            self.make_guess(filter_percent=p)
#            print('Attempting fit with guess {}'.format(self.guess))
            try:
                self.fit_data(force_fit=force_fit)
            except Exception as e:
                self.fit_found = False
                self.amplitude = 0
                self.error = e
                print(e)
            if self.fit_found:
#                print('fit found with {} % filter'.format(p))
                break
            
#    def find_resonance(self):
#        # Pick center frequency based on min and max derivative
#        y = np.real(self.y)
#        window = round(len(y) / 25)  # Average over 2% of the range
#        if (window % 2 == 0):
#            window += 1       # ensure the window has an odd number of points
#        window = max(window, 11)
#        y_smoothed = savgol_filter(y, window, 3)
#        dydx = np.gradient(y_smoothed, self.x)
#        diff = savgol_filter(dydx, window, 3)  # The smoothed derivative
#        #plt.plot(self.x,self.diff);plt.show()
#        #plt.plot(self.x,y2);plt.show()
#        # location should be between the min and max values of the derivative
#        peak_index = int(np.round(np.mean((np.argmax(diff),
#                                   np.argmin(diff)))))
#        location = self.x[peak_index]
    def rotate(self):
        """Try to rotate the complex data such that the real part is Lorentzian
        """
        phi = 0
        y = np.imag(self.y_subset)
        span = self.x_subset[-1] - self.x_subset[0]
#        ymean = np.mean(y)
#        ymax = max(np.abs(self.y_subset))
        yint = np.trapz(y,self.x_subset)/span
        yrange = max(y) - min(y)
        self.rotation_found = True
        while abs(yint) > 0.001*yrange:
#            ymean = np.mean(np.imag(self.y*np.exp(phi*1j)))
            yint = np.trapz(np.imag(self.y_subset*np.exp(phi*1j)),
                            self.x_subset)/span
            phi += 0.001
            if phi > 7:
                self.rotation_found = False
                print("rotation not found")
                break
        
##        y = np.real(self.y*np.exp(phi*1j))
##        self.y = y/max(abs(y))
        if self.rotation_found:
#            print('rotation of {} radians found'.format(phi))
            self.y *= np.exp(phi*1j)
#            self.y = np.real(y/max(np.abs(y)))
        #plt.plot(self.x,self.y);plt.show()
        return None
                
            
    def make_guess(self,filter_percent=2):
        """ToDo:
            *Guess should be made at min/max value inside window definied by
            derivative.  Asymmetry can make it off by quite a bit.
        """
        y = np.real(self.y)
        # Lets make some guesses about reasonable parameters
        self.guess = {}

        # Pick center frequency based on min and max derivative
        window = round(len(y)*0.01*filter_percent)  # Average over % of the range
        if (window % 2 == 0):
            window += 1       # ensure the window has an odd number of points
        window = max(window, 11)
        y_smoothed = savgol_filter(y, window, 3)
#        plt.plot(self.x,y_smoothed);plt.show()
        self.y_smoothed = y_smoothed
        dydx = np.gradient(y_smoothed, self.x)
#        plt.plot(self.x,dydx);plt.show()
        self.diff = savgol_filter(dydx, window, 3)  # The smoothed derivative
#        plt.plot(self.x,self.diff);plt.show()
        #plt.plot(self.x,y2);plt.show()
        # location should be between the min and max values of the derivative
        self.peak_index = np.mean((np.argmax(self.diff),
                                   np.argmin(self.diff)))
        self.peak_index = int(np.round(self.peak_index))
        self.guess['location'] = self.x[self.peak_index]
        
        # use first and last point to define slope and DC offset
        try:
            self.guess['slope'] = (
                (y_smoothed[-1] - y_smoothed[0]) / (self.x[-1] - self.x[0]))
        except Exception as e:  # likely divie by zero warning
            self.guess['slope'] = 0
        # DC offset would be center of the line we use to define slope
        self.guess['DC_offset'] = (y_smoothed[0] + y_smoothed[-1]) / 2
        center = np.mean(self.x)
        # make some compensation for slope, offset
        y2 = y_smoothed - self.guess['DC_offset'] - (
                self.guess['slope']*(self.x-center))
        self.guess['amplitude'] = y2[self.peak_index]
#        plt.plot(self.x,y2);plt.show()

        # Normalize y2
        y2 /= self.guess['amplitude']
#        y2 -= min(y2)
#        y2 /= max(y2)
#        plt.plot(self.x,y2);plt.show()
    
        left_index = self.peak_index
        right_index = self.peak_index
        if y2[self.peak_index] > np.mean(y2):
            while left_index > 0:
                if y2[left_index] > 0.5:
                    left_index -= 1
                else:
                    break
            while right_index < len(y2) - 1:
                if y2[right_index] > 0.5:
                    right_index += 1
                else:
                    break
        else:
            while left_index > 0:
                if y2[left_index] < 0.5:
                    left_index -= 1
                else:
                    break
            while right_index < len(y2) - 1:
                if y2[right_index] < 0.5:
                    right_index += 1
                else:
                    break
        # if the above code worked, make informed guess at FWHM and subsets,
        # else just use a window of 20 points around the expected frequency
        if right_index - left_index >= 20:
            self.guess['gamma'] = (self.x[right_index] - self.x[left_index])/2.
            self.x_subset = self.x[left_index:right_index]
            self.y_subset = self.y[left_index:right_index]
        else:
            right_index = min((self.peak_index + 10,len(self.x)-1))
            left_index = max((self.peak_index - 10,0))
            self.guess['gamma'] = (self.x[right_index] - self.x[left_index])/2.
            self.x_subset = self.x[left_index:right_index]
            self.y_subset = self.y[left_index:right_index]
            

        # Amplitude is guessed at min or max, conditional on kind
##        if self.kind == 'L2':
##            # assume center frequency occurs at max
##            #self.peak_index = np.argmax(y2)
###            while(self.peak_index + 1 < len(self.y) and
###                  y2[self.peak_index + 1] > y2[self.peak_index]):
###                self.peak_index += 1
###            while(self.peak_index - 1 > 0 and
###                  y2[self.peak_index - 1] > y2[self.peak_index]):
###                self.peak_index -= 1
##            self.guess['amplitude'] = max(y2)-min(y2)
##        else:
##            # assume center frequency occurs at min
##            #self.peak_index = np.argmin(y2)
###            while(self.peak_index + 1 < len(self.y) and
###                  y2[self.peak_index + 1] < y2[self.peak_index]):
###                self.peak_index += 1
###            while(self.peak_index - 1 > 0 and
###                  y2[self.peak_index - 1] < y2[self.peak_index]):
###                self.peak_index -= 1
##            self.guess['amplitude'] = min(y2) - max(y2)

        return None

    def fit_data(self, force_fit=False):
        """Fits measured resonator trace to an analytical model.

        Resonator is fit to a lorentzian plus linear background.
        Fit parameters are then updated to object fields

        Returns
        -------
            (Boolean): Success of fit
        """
##        # Find approximate half width
##        arg_center = self.peak_index  # index of peak
##
##        # Make array for slightly cleaner data
##        # make some compensation for slope, offset
###       center = np.mean(self.x)
##        y2 = np.copy(self.y) - self.guess['DC_offset'] - (
##                self.guess['slope']*(self.x))
###        y2 = np.copy(self.y)
##        # Normalize y2
##        y2 -= min(y2)
##        y2 /= max(y2)
##
##
##        # Find approximate half width
##        #arg_center = np.argmin(y2)  # index of peak
##        left_index = arg_center
##        right_index = arg_center
##        
##        #if self.kind == 'L2':
##        if y2[self.peak_index] > np.mean(y2):
##            # search to left
##            while(left_index > 0):
##                if y2[left_index] > 0.5:
##                    left_index -= 1
##                else:
##                    break
##            # search to right
##            while(right_index < len(y2) - 1):
##                if y2[right_index] > 0.5:
##                    right_index += 1
##                else:
##                    break
##        else:
##            # search to left
##            while(left_index > 0):
##                if y2[left_index] < 0.5:
##                    left_index -= 1
##                else:
##                    break
##            # search to right
##            while(right_index < len(y2) - 1):
##                if y2[right_index] < 0.5:
##                    right_index += 1
##                else:
##                    break

        #index_fwhm = (right_index - left_index)

        # Define a window 1 fwhm wide.
#        start = int(arg_center - 2 * index_fwhm)
#        start = start if start >= 0 else 0
#        stop = int(arg_center + 2 * index_fwhm)
#        stop = stop if stop < len(y2) else -1

#        x_subset = self.x[start:stop]
#        y_subset = self.y[start:stop]
##        start = left_index
##        stop = right_index
        #plt.figure()
#        plt.plot(y2)
#        print('index fwhm',index_fwhm)
#        print('start',start)
#        print('stop',stop)
#        plt.figure()

        guess = [self.guess['amplitude'],
                 self.guess['location'],
                 self.guess['gamma'],
                 self.guess['slope'],
                 self.guess['DC_offset']]
        #print(guess)
        
        #convert complex data to real
        y_subset = np.real(self.y_subset)
        y = np.real(self.y)

        try:
            # Fit in restricted range
            pars, covar = curve_fit(Resonator.fit_lorentzian, self.x_subset,
                                    y_subset, guess, xtol=1e-8)
            # Feed resticted fit result in as guess, fit over whole range
            if(self.fit_all):
                guess[1:3] = pars[1:3]
                pars, covar = curve_fit(Resonator.fit_lorentzian, self.x,
                                        y, guess, xtol=1e-8)
        except RuntimeError as e:
            print(e)
            self.fit_found = False
            self.amplitude = 0
            return self.fit_found

        error = np.sqrt(abs(np.diag(covar)))
        self.error['amplitude'] = error[0]
        self.error['location'] = error[1]
        self.error['gamma'] = error[2]
        self.error['slope'] = error[3]
        self.error['DC_offset'] = error[4]
        #print(pars)

        # Lets qualitatively say it is a fit when it has <25% error
        # in the three parameters we care about
        fits_are_good = abs(error[0:3]/pars[0:3]) < 0.15
#        print(error/pars)
        if all(fits_are_good):
            self.fit_found = True
        if pars[1] <= min(self.x) or pars[1] >= max(self.x):
            # resonator center is outside of measured range.
            self.fit_found = False
        if abs(pars[2]) > (self.x[-1]-self.x[0])/2:
            # FWHM is whole trace
            self.fit_found = False
        if force_fit:
            self.fit_found = True

        if self.fit_found or force_fit:
#            print('fit was successful')
            # Update object fields
            self.amplitude = pars[0]
            self.location = pars[1]
            self.gamma = abs(pars[2])
            self.slope = pars[3]
            self.DC_offset = pars[4]

            self.get_fwhm()
            self.get_q()
            self.get_qint()
            self.get_qext()
            self.get_snr()

        if not self.fit_found:
            self.amplitude = 0

        return self.fit_found

    def get_clean_trace(self):
        """Returns raw trace with some DC background removed.

        A function which removes background DC offsets from the resonator
        spectrum. It will not affect point to point variance and therefore
        still includes the original noise.

        Returns:
        --------
            (x,y): numpy.array
                Cleaned resonator specturm
        """
        self.fit_data()
        if self.fit_found is False:
            raise Resonator.FitError("The fit failed. "
                                     "The resonator has no fit parameters.")
        y = np.real(np.copy(self.y))
        x = np.copy(self.x)

        m = self.slope
        b = self.DC_offset
        loc = self.location
        linear_contribution = m * (x-loc) + b
        y -= linear_contribution
        return (x, y)

    def get_fitted_trace(self, frequency=None):
        """Returns pure functional form of resonator with zero noise. Ignores
        bacground slope and DC offset. Used when subtracting a resonator from
        a trace. May not be reliable if fit did not converge.

        Parameters:
            frequency(numpy.array): An array of frequency values over which
            S21 values will be returned.  If not defined, the object's
            orignial trace will be used.

        Returns:
            tuple(numpy.array):
                frequency and S21 data of resonator
        """
        if self.fit_found is False:
            raise Resonator.FitError("The fit failed. "
                                     "The resonator has no fit parameters.")
        if frequency is None:
            frequency = self.x

        signal = Resonator.fit_lorentzian(frequency,
                                          self.amplitude,
                                          self.location,
                                          self.gamma, 0, 0)
        return frequency, signal

    def get_noiseless_trace(self, frequency=None, slope=None):
        """Returns pure functional form of resonator with zero noise. Includes
        bacground slope and DC offset. May not be reliable if fit
        did not converge.

        Parameters:
            frequency(numpy.array): An array of frequency values over which
            S21 values will be returned.  If not defined, the object's
            orignial trace will be used.

        Returns:
            tuple(numpy.array):frequency and S21 data of resonator
        """
        if self.fit_found is False:
            raise BaseException("The fit failed. "
                                "The resonator has no fit parameters.")
        if frequency is None:
            frequency = self.x
        if slope is None:
            slope = self.slope

        signal = Resonator.fit_lorentzian(frequency,
                                          self.amplitude,
                                          self.location,
                                          self.gamma,
                                          slope,
                                          self.DC_offset)
        return frequency, signal

    def fit_lorentzian(x, amp, loc, gamma, slope, DC_offset):
        """Returns a lorentzian for given input

        Parameters
        ----------
            x: numpy.array
                the dependent variable, an array of frequencies
            amp: float
                Amplitude, defining peak height
            loc: float
                The center frequency
            gamma: float
                The half width - half maximum
            slope: float
                Defining the slope of the background
            DC_offset: float
                Defining the offset at the center point of the background

        Returns
        -------
            numpy.array
                y values (S21) to match frequency input x.
        """
        #center = round(len(x)/2)  # The DC offset will be defined at this point
        linear_contribution = 0
        #linear_contribution = slope * (x) + DC_offset  # y = mx + b
        linear_contribution = slope * (x-loc) + DC_offset  # y = mx + b
        lorentzian = amp / ((1 + (((x - loc) / gamma)**2)))
        signal = lorentzian + linear_contribution
        return signal

    def show(self, axes=None,slope=None):
        """Plot the input data and fit

        Returns
        -------
            matplotlib.figure.Figure
                figure handle for plot produced
        """
        import matplotlib.pyplot as plt
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111)
        axes.plot(self.x*1e-9, np.real(self.y), color='blue', label='raw data')
        plt.ylabel('Transmission [S21]', fontname='Arial', fontsize=16)
        plt.xlabel('Frequency [GHz]', fontname='Arial', fontsize=16)
        if self.fit_found:
            xx, yy = self.get_noiseless_trace(slope=slope)
            plt.plot(xx*1e-9, yy, 'r-.', label='fit')
        plt.legend(fontsize=14)
        axes.tick_params(axis='x', labelsize=12)
        axes.tick_params(axis='y', labelsize=12)
        plt.tight_layout()
        # plt.show()
        return fig, axes

    def set_trace(self, x, y):
        assert (len(x) == len(y)), 'X and Y data must be same length!'
        self.x = x
        self.y = y
        self.make_guess()
        self.fit_data()
        return None

    def get_trace(self):
        return (self.x, self.y)

    def get_center(self):
        """Return the resonator's center frequency

        Returns:
            (float): Center frequency [Hz]
        """
        return self.location

    def get_amplitude(self):
        """Return height of resonator above background

        Returns:
            (float): Resonator amplitude
        """
        return abs(self.amplitude)

    def get_gamma(self):
        """Return Lorentzian linewidth gamma

        Returns:
            (float): Resonator linewidth gamma
        """
        return self.gamma

    def get_DC_offset(self):
        """Return DC offset of background

        Returns:
            (float): DC offset of background
        """
        return self.DC_offset

    def get_slope(self):
        """Return background slope

        Returns:
            (float): slope of background
        """
        return self.slope

    def get_fwhm(self):
        """Return the full width at half maximum

        Returns:
            (float): Full width at half Maximum [Hz]
        """
        if not self.fit_found:
            return np.nan
        self.fwhm = 2 * self.gamma
        return (self.fwhm)

    def get_q(self):
        """ Return quality factor of the resonator

        Returns:
            (float): Quality factor (Q)
        """
        if not self.fit_found:
            return np.nan
        self.Q = self.get_center() / self.get_fwhm()
        return self.Q
        
    def get_qint(self):
        if not self.fit_found:
            self.Qint = np.nan
            return self.Qint
        norm_amp = self.get_amplitude()/abs(self.get_DC_offset())
        self.Qint = self.get_center() * self.get_fwhm()/(self.get_fwhm()**2 - 2*norm_amp*self.get_gamma()**2)
        return self.Qint
        
    def get_qext(self):
        if not self.fit_found:
            self.Qext = np.nan
            return self.Qext
        norm_amp = self.get_amplitude()/abs(self.get_DC_offset())
        self.Qext = self.get_center() * self.get_fwhm()/(2*norm_amp*self.get_gamma()**2)
        return self.Qext
        
    def get_ringdown_time(self):
        """"Return ringdown time, t_1/2 = 2 ln(2) Q/omega
        Returns:
            (float): Ringdown time in seconds
        """
        if not self.fit_found:
            return np.nan
        self.ringdown_time = (np.log(2)/np.pi) * self.Q / self.location
        return self.ringdown_time

    def get_snr(self):
        """Compute signal to noise ratio.  Useful for determining how long to
        average, or if this is the strongest resonator found in a set.
        I imagine a case where you look for all resonance peaks in a trace
        and sort them by their SNR to distinguish between smaller spurious
        modes and larger intended resonators.

        Returns:
            (float): Signal to noise ratio.  -1 if data cannot be fit.
        """
        if not self.fit_found:
            return np.nan

        f, signal = self.get_noiseless_trace()
        noise = abs(np.copy(self.y) - signal)
        noise_amplitude = np.mean(noise)
        signal_amplitude = (self.amplitude)
        self.snr = (signal_amplitude/noise_amplitude)**2
        return self.snr

    def get_frac_error(self):
        """return a dictionary of the fractional errors in the fit values"""

        d = {}
        d['amplitude'] = self.error['amplitude'] / self.amplitude
        d['location'] = self.error['location'] / self.location
        d['gamma'] = self.error['gamma'] / self.gamma
        d['slope'] = self.error['slope'] / self.slope
        d['DC_offset'] = self.error['DC_offset'] / self.DC_offset
        return d

    def search(x, y, kind='L4', number=-1, ignore=[]):
        """Return a list of all resonators found in a trace.

        Parameters
        ----------
            x(np.array):
                frequency data for trace
            y(np.array):
                S21 data for trace
            kind(str):
                L2 or L4 resonator type
            number(int):
                Total number of resonators to return.  Entering a
                negative number will return all resonators with an SNR > 1.
            ignore(list of Resonators):
                If these are found, they will not be added to the output

        Returns
        -------
            (list): A list of Resonators, in order of highest peak
        """
        result = []
        # in case we are handed a single object
        if isinstance(ignore, Resonator):
            ignore = [ignore]
        yc = np.copy(y)
        while len(result) < number or number < 0:
            res = Resonator(x=x, y=yc, kind=kind)
            # quit if we start getting garbage
            if not res.fit_found or res.get_snr() < 1:
                break
            # Add to list
            if res not in ignore and res not in result:
                result.append(res)
            # remove it from the trace
            _, yf = res.get_fitted_trace()
            yc -= yf
        num_found = (len(result))
        print('Found ' + str(num_found) + ' resonator' + ('s' *
              (num_found != 1)) + '.')
        return result

    def __repr__(self):
        return ('%.3f' % (self.get_center() / 1e9) + ' GHz')

    def __str__(self):
        """Return formatted string containing all extracted values
        """

        if not self.fit_found:
            return 'Data was not fitted.'
        frac_error = self.get_frac_error()
        string = (
                'Amplitude:        ' + '%9.3e' % self.amplitude +
                ' +/- ' + '%5.2f' % (100*frac_error['amplitude']) + ' %' +
                '\nLocation:       ' + '%9.6e' % self.location +
                ' +/- ' + '%5.2e' % (100*frac_error['location']) + ' % [Hz]' +
                '\nGamma:          ' + '%9.6e' % self.gamma +
                ' +/- ' + '%5.2f' % (100*frac_error['gamma']) + ' %' +
                '\nSlope:          ' + '%9.3e' % self.slope +
                ' +/- ' + '%5.2f' % (100*frac_error['slope']) + ' %' +
                '\nDC Offset:      ' + '%9.3e' % self.DC_offset +
                ' +/- ' + '%5.2f' % (100*frac_error['DC_offset']) + ' %' +
                '\nFWHM:           ' + '%9.3e' % self.fwhm +
                ' +/- ' + '%5.2f' % (100*frac_error['gamma']) + ' % [Hz]' +
                '\nSNR             ' + '%9.1f' % self.snr +
                ' +/- ' + '%5.2f' % (100*frac_error['amplitude']) + ' %' +
                '\nQ:              ' + '%9.1f' % self.Q +
                ' +/- ' + '%5.2f' % (100 * frac_error['location'] +
                                     frac_error['gamma']) + ' %' +
                '\nQ internal:     ' + '%9.1f' % self.Qint +
                ' +/- ' + '%5.2f' % (100 * frac_error['location'] +
                                     frac_error['gamma']) + ' %' +
                '\nQ external:     ' + '%9.1f' % self.Qext +
                ' +/- ' + '%5.2f' % (100 * frac_error['location'] +
                                     frac_error['gamma']) + ' %' +
                '\nRingdown Time:  ' + '%9.3f' % (1e6 * 4.53 *
                                                  self.Q / self.location) +
                ' +/- ' + '%5.2f' % (100*(frac_error['location'] +
                                     frac_error['gamma'])) + ' % [us]'
                )

        return string

    def __eq__(self, other):
        """define equality as having centers within
        one FWHM of each other"""
        # Identity check
        if self is other:
            return True
        # Type check
        if not type(self) == type(other):
            return False
        # If they overlap they are equal
        if (abs(self.location - other.location) <
                max(self.gamma, other.gamma)):
            return True
        return False

    def __ne__(self, other):
        """Equality is defined as having centers within
        one FWHM of each other"""
        # Identity check
        if self is other:
            return False
        # Type check
        if not type(self) == type(other):
            return True
        # If they overlap they are equal
        if (abs(self.location - other.location) <
                max(self.gamma, other.gamma)):
            return False
        return True

    def __lt__(self, other):
        return(self.location < other.location)

    def __gt__(self, other):
        return(self.location > other.location)

    def __le__(self, other):
        return (self < other or self == other)

    def __ge__(self, other):
        return (self > other or self == other)


class peaks:
    """this will search through a logfile and looks for peaks in each trace.
    call get_peaks() to get a lists of
    """

    def __init__(self, filepath):
        t0 = time()
        self.filepath = filepath
        lf = LogFile.LogFile(filepath)
        self.hasTraces = lf.hasTraces()
        lf.close()
        if self.hasTraces:
            self.load_trace_data()
        else:
            self.load_digitizer_data()
        self.clean_data()
        self.find_resonators()
        self.plot2D()
#        self.plot_contour()
#        self.plot_3D()
        print('Script complete in ', '%.2f' % (time()-t0), ' seconds')
        gc.collect()

    def load_digitizer_data(self):
        lf = LogFile.LogFile(self.filepath)
        self.step_channels = lf.getStepChannels()
        self.log_channels = lf.getLogChannels()
        self.freq_name = self.step_channels[0].get('name')
        self.bias_name = self.step_channels[1].get('name')
        self.value_name = self.log_channels[0].get('name')
        if ('frequency' not in self.freq_name.lower() and
            'frequency' not in self.bias_name.lower()):
            raise AttributeError('File does not contain frequency-dependent '
                                 'data.')
        # Get data
        self.freq_data = lf.getData(self.freq_name)
        self.bias_data = lf.getData(self.bias_name)
        self.values = lf.getData(self.value_name)

        if 'frequency' in self.bias_name.lower():
            # Data was taken in wrong order.  Swap them.
            self.freq_name = self.step_channels[1].get('name')
            self.bias_name = self.step_channels[0].get('name')
            self.freq_data = lf.getData(self.freq_name).transpose()
            self.bias_data = lf.getData(self.bias_name).transpose()
            self.values = lf.getData(self.value_name).transpose()

        self.bias_points = self.bias_data[0, :]
        self.shape = np.shape(self.freq_data)
        self.n_entries = np.shape(self.freq_data)[1]
        self.f = self.freq_data[:,  0]
        self.S21_mat = [self.values[:, x] for x in range(self.n_entries)]
        self.S21_mat = abs(np.array(self.S21_mat))
        lf.close()
        return None

##    def load_trace_data(self):
##        """Currently not sure if this works.  Was designed for a 3D log file
##        [JB v1.0.7]
##        """
##        print('Loading Data....   ', end='')
##        t0 = time()
##        lf = Labber.LogFile(self.filepath)
##        self.step_channels = lf.getStepChannels()
##        self.log_channels = lf.getLogChannels()
##        self.freq_name = self.step_channels[0].get('name')
##        self.freq_data = lf.getData(self.freq_name)
##        self.bias_name = self.step_channels[1].get('name')
##        self.bias_data = lf.getData(self.bias_name)
##        self.n_entries = np.shape(self.freq_data.flat)[0]
##
##        self.f, _ = lf.getTraceXY()
##
##        # Explicitly delete logfile since it has no close function
##        del(lf)
##        gc.collect()
##
##        # Get S21 data using acquisition (takes 1 second)
##        with LogFile.LogFile(self.filepath, 'r') as lf:
##            self.S21_mat = abs(lf.getTraceXY()[1].transpose())
##
##        t1 = time()
##        print('Finished in ' + '%.3f' % (t1-t0) + ' seconds')
##        return None

    def clean_data(self):
        """For all traces, subtract median value at that frequency.
        Much better than subtracting mean, which may leave ghost features
        near max/min frequencies in a sweep"""
        # subtract out the background by subtracting average of
        # slices at same freq
        t0 = time()
        for n in tqdm(range(len(self.f))):
            self.S21_mat[:, n] -= np.median(self.S21_mat[:, n])
        print('\rCleaning data...   Finished in ' +
              '%.3f' % (time()-t0) + ' seconds')
        return None

    def find_resonators(self, min_amplitude=None):
        """ Create a matrix of the resonant frequency of each trace """
        t0 = time()
        self.fmax = np.zeros(self.n_entries)
        self.amplitudes = np.zeros(self.n_entries)
        self.res_mat = [None] * self.n_entries

        for n in tqdm(range(self.n_entries)):
            try:
                res = Resonator(self.f, self.S21_mat[n], force_fit=False)
                self.res_mat[n] = res
                self.amplitudes[n] = res.get_amplitude()
                if res.fit_found is False:
                    print('\rTrace ', n, ' did not fit.', end='')
            except Exception as e:
                print('\rTrace ', n, ' did not fit.', end='')

        # Need to define a minimum resonator amplitude, below which
        # it is assumed to be junk and ignored.  What is a low amplitude?
        # Lets parse through the data and see.
        # First: collect all amplitude values
        self.amplitudes = np.empty(len(self.res_mat))
        for n, r in enumerate(self.res_mat):
            self.amplitudes[n] = r.get_amplitude()

        # Screen out all of the zero values (meaning resonator not found)
        df = pd.DataFrame(self.amplitudes)
        non_zeros = df.loc[(df != 0).any(axis=1)]
        non_zeros = non_zeros.as_matrix().tolist()
        self.non_zeros = np.array(non_zeros)
        self.mean_amplitude = np.mean(non_zeros)
        self.std_amplitude = np.std(non_zeros)

        # Use the derived amplitudes to define the cutoff, allow override
        # for special cases
        if not min_amplitude:
            amp = (abs(self.mean_amplitude) - 2*self.std_amplitude)
            min_amplitude = max(amp, abs(0.2 * self.mean_amplitude))

        self.min_amplitude = min_amplitude
        self.fmax = []
        self.final_bias_points = []

        for n, res in enumerate(self.res_mat):
            fres = res.get_center()
            if abs(res.get_amplitude()) < min_amplitude or not res.fit_found:
                fres = self.f[0]
                if not res.fit_found:
                    # print(n, ": didn't fit")
                    continue
                elif abs(res.get_amplitude()) < min_amplitude:
                    # print(n, ': too small')
                    continue
            self.fmax.append(fres)
            self.final_bias_points.append(self.bias_points[n])

        self.res_mat = np.array(self.res_mat)
        print('\rFitting data....   Finished in ' +
              '%.3f' % (time()-t0) + ' seconds')
        return None

    def plot_contour(self, title=None, xlabel=None, ylabel=None):
        plt.figure()
        cplt = plt.contourf(self.freq_data, self.bias_data,
                            self.fmax, 128, cmap='jet')
        cbar = plt.colorbar()
        cbar.set_label('Frequency [Hz]')
        if not title:
            title = os.path.basename(self.filepath)
        if not xlabel:
            xlabel = self.freq_name
        if not ylabel:
            ylabel = self.bias_name
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        fig_path = self.filepath.split('.hdf5')[0] + '.jpg'
        plt.savefig(fig_path, dpi=600)
        print('Figure saved to: ', fig_path)
        return cplt

    def plot_3D(self, angle=-85):
        fig3D, ax3D = plt.subplots(subplot_kw=dict(projection='3d'),
                                   figsize=(11, 8))
        xi = np.linspace(self.freq_data.min(),
                         self.freq_data.max(), len(self.fmax[0]))
        yi = np.linspace(self.bias_data.min(), self.bias_data.max(),
                         len(self.fmax))
        X, Y = np.meshgrid(self.freq_data[0], self.bias_data[:, 0])
        # define a grid, a bit differently this time.
        # We will use these to define the granularity of the plot
        rows = len(self.freq_data[0])
        cols = len(self.bias_data[:, 0])

        # Map your data onto the grid
        zi = griddata(self.freq_data.flatten(), self.bias_data.flatten(),
                      abs(self.fmax).flatten(), xi, yi, interp='linear')
        surf = ax3D.plot_surface(X, Y, zi, vmin=abs(self.fmax).min(),
                                 vmax=abs(self.fmax).max(), cmap='jet',
                                 rcount=rows, ccount=cols)
        ax3D.set_zlim(abs(self.fmax).min(), abs(self.fmax).max())
        ax3D.view_init(40, angle)  # Here we define the viewing angle
        ax3D.set_title('Resonator Frequency')
        ax3D.set_xlabel(self.freq_name)
        ax3D.set_ylabel(self.bias_name)

        fig_path = self.filepath.split('.hdf5')[0] + '_3D.jpg'
        plt.savefig(fig_path, dpi=600)
        print('Figure saved to: ', fig_path)

        return surf

    def hist_amplitude(self):
        plot = plt.hist(abs(self.non_zeros), bins=11, label='Counts')
        plt.plot([self.min_amplitude, self.min_amplitude],
                 [0, max(plot[0])], label='Cutoff', linewidth=3)
        plt.xlabel('Amplitude (magnitude)')
        plt.ylabel('Counts')
        plt.legend()
        return plot

    def plot2D(self):
        fig = plt.figure()
        plt.plot(self.final_bias_points, self.fmax, 'rx--')
        plt.title('Resonator Frequency')
        plt.xlabel(self.freq_name)
        plt.ylabel(self.bias_name)
        plt.ylim(min(self.f), max(self.f))
        return fig

    def show_result_over_raw(self):
        fig = plt.figure()
        plt.plot(self.final_bias_points, self.fmax, 'rx--')
        plt.title('Resonator Frequency')
        plt.xlabel(self.freq_name)
        plt.ylabel(self.bias_name)
        plt.ylim(min(self.f), max(self.f))
        std = np.std(self.S21_mat)
        mean = np.mean(self.S21_mat)
        plt.pcolor(self.bias_data.transpose(),
                   self.freq_data.transpose(), self.S21_mat,
                   vmin=mean-3*std, vmax=mean+3*std)
        plt.colorbar()
        return fig



#    def screen_results(self):
#        res_final = []
#        res_freqs = []
#        freq_final = []
#        bias_final = []
#
#        for n, r in enumerate(self.res_mat()):
#            if r.get_snr() > 5 and r.get_fwhm() < (f[-1] - f[0])/10:
#                res_final.append(r)
#                res_freqs.append(r.get_center())
#                freq_final.append()
