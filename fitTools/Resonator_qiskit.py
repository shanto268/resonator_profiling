# -*- coding: utf-8 -*-
"""
A resonator object which fits complex scattering data to the model for direct reflection, direct transmission, or notch
type resonators. If desired, one can use high snr data to first calibrate the overall environmental effects such as phase delay and rotation with resonator.add_calibration_data(f_data,z_data). Subsequent data added with resonator.new_data(f_data,z_data) will be normalized by the overall calibration found previously.

The circlefit method is adapted from Probst et al. 'Efficient and robust analysis of complex scattering data under noise in microwave resonators' (2015)

"""

from fitTools.utilities import plotting, save_load, Watt2dBm, dBm2Watt
from fitTools.circlefit import circlefit
from fitTools.calibration import calibration

from scipy.constants import hbar
from scipy.signal import savgol_filter
import numpy as np 
import warnings
import scipy.optimize as spopt

##
## z_data_raw denotes the raw data
## z_data denotes the normalized data
##

class Resonator(circlefit, save_load, plotting, calibration):
    
    def __init__(self,port_type='reflection',f_data=None, z_data=None):
        '''A resonator fitting object. port_type must be a string identifying the type
        of resonator geometry as "reflection", "transmission", or "notch"
        '''
        if type(port_type) is not str:
            print('WARNING: port_type provided was not a string. Assuming a reflection geometry.')
            self.port_type = 'R'
        else:
            if port_type.lower()[0] == 'r':
                self.port_type = 'R'
            elif port_type.lower()[0] == 't':
                self.port_type = 'T'
            elif port_type.lower()[0] == 'n':
                self.port_type = 'N'
            else:
                print('WARNING: port_type provided could not be understood. Assuming a reflection geometry.')
                self.port_type = 'R'
        self.z_data = None
        self.calibration_data = None
        if f_data is not None:
            self.f_data = np.array(f_data)
        else:
            self.f_data = None
        if z_data is not None:
            self.z_data_raw = np.array(z_data)
        else:
            self.z_data_raw = None
            
        # define parameters as nans, update with fit results later
        self.f0 = np.nan
        self.Q = np.nan
        self.Qi = np.nan
        self.Qc = np.nan
        self.absQc = np.nan
        self.kappa = np.nan
        self.snr = np.nan
        self.diameter = np.nan
        self.delay = np.nan
        self.ringdown_time = np.nan
        self.theta0 = np.nan
        self.offrespoint = np.nan
        self.alpha = np.nan
        self.amp = np.nan
        self.phi0 = np.nan
        self.Q_est = np.nan
        self.f0_est = np.nan
        self.P1photon = np.nan
        self.amp_trans = np.nan
        
        self.fit_found = False
        self.calibrated = False
        self.normalized = False
        
    def autofit(self,electric_delay=None,fcrop=None,force_calibrate=False):
        '''
        automatic normalization and fitting.
        electric_delay: set the electric delay manually, not recommended.
        fcrop = (f1,f2) in GHz: crop the frequency range used for fitting.
        calibration of environment effects will be performed if it has not yet been 
        done or it is forced with the force_calibrate option.
        '''
        if fcrop is None:
            self._fid = np.ones(self.f_data.size,dtype=bool)
        else:
            f1, f2 = np.array(fcrop) * 1e9
            self._fid = np.logical_and(self.f_data>=f1,self.f_data<=f2)
        
        # There is no extra information to be gained by circlefit or phase fit for transmission resonators,
        # so the fit methods are different. Handle notch and resonators first
        if self.port_type is not 'T':
            if force_calibrate or not self.calibrated:
                self.do_calibration(ignore_slope=True,guess_delay=True,
                                    fixed_delay=electric_delay,force_calibrate=force_calibrate)
            
            # perform the canonical normalization that rotates the off resonant point to the real axis at +- 1. 
            self.do_normalization()
        
            # run circlefit with the (possibly) cropped, canonically normalized data. 
            # self.f0_est and self.Q_est used as initial guesses for phasefit
            self.circlefit(refine_results=False,calc_errors=True)
        else:
            self.fit_transmission()
        
        # generate the fitted model data
        if self.fit_found:
            if self.port_type is 'R':
                self.z_data_sim = self._S11_directrefl(self.f_data,fr=self.f0,Ql=self.Q,Qc=self.Qc,
                                                       a=self.amp,alpha=self.alpha,delay=self.delay)
                self.z_data_sim_norm = self._S11_directrefl(self.f_data,fr=self.f0,Ql=self.Q,Qc=self.Qc,
                                                            a=1.,alpha=0.,delay=0.)
            elif self.port_type is 'N':
                self.z_data_sim = self._S21_notch(self.f_data,fr=self.f0,Ql=self.Q,Qc=self.absQc,phi=self.phi0,
                                                  a=self.amp,alpha=self.alpha,delay=self.delay)
                self.z_data_sim_norm = self._S21_notch(self.f_data,fr=self.f0,Ql=self.Q,Qc=self.absQc,phi=self.phi0,
                                                            a=1.,alpha=0.,delay=0.)
            elif self.port_type is 'T':
                self.z_data_sim = self._S21_direct(self.f_data,fr=self.f0,Ql=self.Q,A=self.amp_trans,alpha=self.alpha,
                                               delay=self.delay)
                self.z_data_sim_norm = self._S21_direct(self.f_data,fr=self.f0,Ql=self.Q,A=self.amp_trans,alpha=0,delay=0)
                
            self.snr = self.get_snr()
        else:
            print('The fit could not be found, try cropping the data with autofit(fcrop=(f1 [GHz],f2 [GHz]))')
        
    def do_calibration(self,ignore_slope=True,guess_delay=True,fixed_delay=None,force_calibrate=False):
        '''
        performs an automated calibration and tries to determine the prefactors a, alpha, delay
        fr, Ql, and a possible slope are extra information, which can be used as start parameters for subsequent fits
        see also "do_normalization"
        the calibration procedure works for transmission line resonators as well
        '''
        # if no calibration data is present or the force_calibrate option is given, then use the current dataset
        if force_calibrate or self.calibration_data is None:
            z_data = self.z_data_raw[self._fid]
            f_data = self.f_data[self._fid]
        else:
            z_data = self.calibration_data['z']
            f_data = self.calibration_data['f']
        
        # normalize the complex z data by it's max magnitude
        #maxval = np.max(np.absolute(z_data))
        #z_data_temp = z_data/maxval
        
        # fit the electrical delay in the data, unless the delay is provided already.
        #if type(fixed_delay) not in (int,float,np.float32,np.float64,np.int):
        #    if guess_delay == True:
        #        delay = self._guess_delay(f_data,z_data_temp)
        #    else:
        #        delay=0.
        #    self.delay = self._fit_delay(f_data,z_data_temp,delay,maxiter=200)
        #else:
        #    self.delay = fixed_delay

        self.delay,params = self.get_delay(f_data,z_data,ignoreslope=ignore_slope,guess=guess_delay,delay=fixed_delay)
            
        # remove electrical delay
        z_data *= np.exp(2j*np.pi*self.delay*f_data)
        
        # fit the circle in complex plane
        xc, yc, r0 = self._fit_circle(z_data)
        zc = np.complex(xc,yc)
        
        # center the circle at the complex origin
        z_data_center = self._center(z_data,zc)
        
        # smooth the phase of centered data
        #window = round(len(z_data)*0.01)  # Average over 1% of the range
        #if (window % 2 == 0):
        #    window += 1       # ensure the window has an odd number of points
        #window = max(window, 11)
        #phase_smoothed = savgol_filter(np.angle(z_data_center), window, 3)
        # plot for debugging purposes
        #import matplotlib.pyplot as plt
        #plt.plot(f_data,phase_smoothed)
        #plt.show()
        
        # estimate the resonant frequency by taking the gradient of phase. 
        # this estimation method is robust to arbitrary phase offset and branch cuts.
        #phase_data = np.angle(z_data_center)
        #grad = np.abs(np.gradient(np.sin(phase_data)))
        #fr_ind = np.argmax(grad)
        #f0_est = f_data[fr_ind]
        #print('gradient estimate of f0: {:.3f} GHz'.format(f0_est*1e-9))
        #plt.plot(f_data,grad)
        #plt.show()
        
        # estimate the bandwidth with gradient of phase at resonance
        #slope = np.gradient(phase_data)[fr_ind]
        #Q_est = abs(slope*f0_est/(1000*np.pi))
        #print('Q_est = {:.1f}'.format(Q_est))
        #bw_ind = np.logical_and(phase_data < (phase_data[fr_ind]+np.pi/2),
        #                        phase_data > (phase_data[fr_ind]-np.pi/2))
        #bw = f_data[bw_ind]
        #Q_est = 2*np.pi*f0_est/(bw[-1]-bw[0])
        #print('Q_est is {}'.format(Q_est))
        
        # estimate the bandwidth from gradient method used above.
        #res_id = grad_tan > 1
        #if not res_id.any():
        #    print('bandwidth estimation failed. Using a standard guess of 30 elements.')
        #    start = max(0,fr_ind-15)
        #    stop = min(len(res_id),fr_ind+15)
        #    res_id[start:stop] = True
        #res_f = f_data[res_id]
        #Q_est = f_data[fr_ind]/(res_f[-1]-res_f[0])
        

        # fit the phase of centered circle. theta is the phase offset
        #print('pre-estimates are f0: {}, Q: {}'.format(f0_est,Q_est))
        theta, self.Q_est, self.f0_est = self._phase_fit(f_data,z_data_center,0,np.absolute(params[5]),params[4])
        
        # beta is the negative supplement of theta, it is the angle of the offresonant point relative to center of circle.
        beta = self._periodic_boundary(theta+np.pi,np.pi)
        
        # construct the off resonant point from circle center/radius and angle beta
        self.offrespoint = np.complex((xc+r0*np.cos(beta)),(yc+r0*np.sin(beta)))
        
        # alpha is the phase of offresonant point. it's the angle you must rotate the data so off resonance is purely real.
        self.alpha = np.angle(self.offrespoint) if self.port_type is 'N' else self._periodic_boundary(
            np.angle(self.offrespoint)+np.pi, np.pi)
        
        # amp is the magnitude of the off resonant point. this is used to scale the data so off resonance is 1.
        self.amp = np.absolute(self.offrespoint)
        
        # the impedance mismatch can be found as the angle between vectors from offrespoint to origin
        # and offrespoint to center of circle. We know the length of both vectors and the length of vector
        # from origin to center of circle, so use law of cosines to find this angle.
        self.phi0 = -np.arccos((np.absolute(zc**2) - r0**2 - self.amp**2)/(-2*r0*self.amp))
        #print('r0: {0:.3f}, zc: {1:.3f}, a: {2:.3f}, phi0: {3:.3f}'.format(r0,np.absolute(zc),a,self.phi0))
        
        self.calibrated = True
        #print('Estimates from cal\nphi0: {:.3f},f0: {:.3f},Q: {:.1f}'.format(self.phi0,self.f0_est,self.Q_est))
        return
    
    def get_delay(self,f_data,z_data,delay=None,ignoreslope=True,guess=True):
        '''
        retrieves the cable delay assuming the ideal resonance has a circular shape
        modifies the cable delay until the shape Im(S21) vs Re(S21) is circular
        see "do_calibration"
        '''
        maxval = np.max(np.absolute(z_data))
        z_data = z_data/maxval
        A1, A2, A3, A4, fr, Ql = self._fit_skewed_lorentzian(f_data,z_data)
        if ignoreslope==True:
            A2 = 0.
        else:
            A2 = 0.
            print("WARNING: The ignoreslope option is ignored! Corrections to the baseline should be done manually prior to fitting.")
            print("see also: resonator_tools.calibration.fit_baseline_amp() etc. for help on fitting the baseline.")
            print("There is also an example ipython notebook for using this function.")
            print("However, make sure to understand the impact of the baseline (parasitic coupled resonances etc.) on your system.")
            #z_data = (np.absolute(z_data)-A2*(f_data-fr)) * np.exp(np.angle(z_data)*1j)  #usually not necessary
        if delay is None:
            if guess==True:
                delay = self._guess_delay(f_data,z_data)
                #print('guessed delay is {}'.format(delay))
            else:
                delay=0.
            delay = self._fit_delay(f_data,z_data,delay,maxiter=200)
        params = [A1, A2, A3, A4, fr, Ql]
        return delay, params
        
    def do_normalization(self):
        '''
        removes the prefactors a, alpha, delay and returns the calibrated data, see also "do_calibration"
        '''
        if not self.calibrated:
            print('WARNING: calibration must be performed first by calling Resonator.do_calibration() or Resonator.fit()')
            print('Aborting normalization.')
            self.normalized = False
        else:
            self.z_data = self.z_data_raw*np.exp(1j*(-self.alpha+2.*np.pi*self.delay*self.f_data))/self.amp
            self.normalized = True
        return
        
    def circlefit(self,refine_results=False,calc_errors=True):
        '''
        performs a circle fit on a frequency vs. complex resonator scattering data set
        Data has to be normalized!!
        INPUT:
        f_data,z_data: input data (frequency, complex scattering data)
        OUTPUT:
        no direct output, but fit parameters are saved as class variables accesible elsewhere.
        for details, see:
            [1] (not diameter corrected) Jiansong Gao, "The Physics of Superconducting Microwave Resonators" (PhD Thesis), Appendix E, California Institute of Technology, (2008)
            [2] (diameter corrected) M. S. Khalil, et. al., J. Appl. Phys. 111, 054510 (2012)
            [3] (fitting techniques) N. CHERNOV AND C. LESORT, "Least Squares Fitting of Circles", Journal of Mathematical Imaging and Vision 23, 239, (2005)
            [4] (further fitting techniques) P. J. Petersan, S. M. Anlage, J. Appl. Phys, 84, 3392 (1998)
            [5] Probst et al. Efficient and robust analysis of complex scattering data under noise in microwave resonators' (2015)
        the program fits the circle with the algebraic technique described in [3], the rest of the fitting is done with the scipy.optimize least square fitting toolbox
        '''
        
        # make sure the data has been calibrated and normalized.
##        if not self.calibrated:
##            print('WARNING: calibration must be performed first by calling Resonator.do_calibration() or Resonator.fit()')
##            print('Aborting circlefit.')
##            return
##        if not self.normalized:
##            print('WARNING: normalization must be performed first by calling Resonator.do_normalization() or Resonator.fit()')
##            print('Aborting circlefit.')
##            return
        
        # fit the circle in complex plane
        xc, yc, r0 = self._fit_circle(self.z_data,refine_results=refine_results)
        
        # since the data has already been normalized, offrespoint lies at 1 on the real axis and a 
        # right triangle is formed with between circle center, offrespoint, and real axis. 
        # then the impedance mismatch angle can be found again as below:
        self.phi0_2 = -np.arcsin(yc/r0)
        
        # estimate the phase offset as supplement to impdance mismatch, assuming that the circle will be centered at origin
        theta0 = self._periodic_boundary(self.phi0+np.pi,np.pi)
        z_data_center = self._center(self.z_data,np.complex(xc,yc))
        
        # iteratively fit the phase of centered data
        theta0, Ql, fr = self._phase_fit(self.f_data,z_data_center,theta0,self.Q_est,self.f0_est)
        #print('phase fit results\ntheta0: {:.2f},Q: {:.1f},f0: {:.3f}'.format(theta0,Ql,fr))
        
        # once again, impedance mismatch phi0 is supplement to phase offset in the centered, normalized frame
        self.phi0_3 = self._periodic_boundary(theta0 + np.pi,np.pi) if self.port_type is not 'R' else theta0
        # let's just take the average of the three ways of finding this thing and make sure they don't disagree by much:
        lphi0 = [abs(self.phi0),abs(self.phi0_2),abs(self.phi0_3)]
        #print('the phi estimates:{}'.format(lphi0))
        if np.std(lphi0) > 1e-8:
            #print('the phi estimates:{}'.format(lphi0))
            #print('Disagreement in impedance mismatch. Using value from canonical position.')
            phi0 = self.phi0_2
        else:
            phi0 = -np.mean(lphi0)
        
        # now solve for the quality factors depending on port type
        cosphi = np.cos(phi0)
        if self.port_type == 'R':
            absQc = Ql/r0
            Qc = Ql/(r0*cosphi)
            Qi = Ql/(1-r0*cosphi)
        else:
            absQc = Ql/(2*r0)
            Qc = Ql/(2*r0*cosphi)
            Qi = Ql/(1-2*r0*cosphi)
        #print('Qi: {:.1f}'.format(Qi))
        
        #complQc = absQc*np.exp(1j*((-1.)*phi0))
        #Qc = 1./(1./complQc).real	# here, taking the real part of (1/complQc) from diameter correction method
        #Qi_dia_corr = 1./(1./Ql-1./Qc)
        #Qi_no_corr = 1./(1./Ql-1./absQc)
        #Qc = Ql/(2*r0*np.cos(phi0)) # this is the real part of complex Qc
        #Qi_dia_corr = Ql/(1-2*r0*np.cos(phi0)) # this is with diameter correction
        #Qi_no_corr = Ql/(1-2*r0) # without diameter correction

        # calculation of the error
        
        p = [fr,absQc,Ql,phi0] if self.port_type is not 'R' else [fr,absQc,Ql]
        #chi_square, errors = rt.get_errors(rt.residuals_notch_ideal,f_data,z_data,p)
        if calc_errors==True:
            chi_square, cov = self._get_cov_fast_notch(self.f_data,self.z_data,p) if self.port_type is not 'R' else self._get_cov_fast_directrefl(self.f_data,self.z_data,p)
            #chi_square, cov = rt.get_cov(rt.residuals_notch_ideal,f_data,z_data,p)
        
            if cov is not None:
                errors = np.sqrt(np.diagonal(cov))
                if self.port_type is not 'R':
                    fr_err,Qc_err,Ql_err,phi0_err = errors
                    #calc Qi dia corr with error prop
                    dQl = 1/((1/Ql-np.cos(phi0)/absQc)**2 *Ql**2)
                    dabsQc = -np.cos(phi0)/((1/Ql-np.cos(phi0)/absQc)**2 *absQc**2)
                    dphi0 = -np.sin(phi0)/((1/Ql-np.cos(phi0)/absQc)**2 *absQc)
                    ##err1 = ( (dQl*cov[2][2])**2 + (dabsQc*cov[1][1])**2 + (dphi0*cov[3][3])**2 )
                    err1 = ( (dQl**2*cov[2][2]) + (dabsQc**2*cov[1][1]) + (dphi0**2*cov[3][3]) )
                    err2 = ( dQl*dabsQc*cov[2][1] + dQl*dphi0*cov[2][3] + dabsQc*dphi0*cov[1][3] )
                    Qi_err =  np.sqrt(err1+2*err2)	 # including correlations
                else:
                    fr_err,Qc_err,Ql_err = errors
                    #calc Qi with error prop (sum the squares of the variances and covariaces)
                    dQl = 1./((1./Ql-1./absQc)**2*Ql**2)
                    dQc = - 1./((1./Ql-1./absQc)**2*absQc**2)
                    Qi_err = np.sqrt((dQl**2*cov[2][2]) + (dQc**2*cov[1][1])+(2*dQl*dQc*cov[2][1]))	 #with correlations

                # call the fit good if there's less that 20% error in the 3 parameters we care most about
                errors = np.array([fr_err,Qc_err,Ql_err])
                params = np.array([fr,absQc,Ql])
                fits_are_good = np.abs(errors/params) < 0.3 
                if all(fits_are_good):
                    self.fit_found = True
                else:
                    print('fit error = {}'.format(errors/params))
                    self.fit_found = False

            else:
                print("WARNING: Error calculation failed!")
                self.fit_found = False
        else:
            #just calc chisquared:
            fun2 = lambda x: self._residuals_notch_ideal(x,f_data,z_data)**2
            chi_square = 1./float(len(f_data)-len(p)) * (fun2(p)).sum()
            if chi_square > 0.85:
                self.fit_found = True
            else:
                self.fit_found = False

        # update variables if the fit was good.
        if self.fit_found:
            self.f0 = fr
            self.Q = Ql
            self.Qi = Qi
            self.Qc = Qc
            self.absQc = absQc
            self.phi0 = phi0
            self.theta0 = theta0
            self.kappa = 2*np.pi*fr/Ql
            self.diameter = 2*r0
            self.amplitude = 2*r0*cosphi
            self.ringdown_time = (np.log(2)/np.pi) * self.Q / self.f0
            self.theta0 = theta0
            self.P1photon = self.get_single_photon_limit()
            
    def fit_transmission(self):
        '''
        A method for fitting the direct transmission resonance. This one is tricky.
        The ideal transmission resonator has |S21|^2 = Lorentzian, with off resonance ~= zero.
        However, simply trying to fit the amplitude squared data to a lorentzian does not work well
        because the noise away from resonance is no longer centered about zero, but some small positive offset.
        The positive offset applied only to the tails around the resonance makes curve fitting methods fail.
        To get around this, we use the circle fit method to remove electrical delay, center complex data, and fit
        the phase response. This provides us with the resonant frequency and quality factor. After that, we can use
        f0 and Q to properly fit the amplitude^2 to the lorentzian model in a restricted subset of the data where
        noise does not skew the result.
        '''
        #import matplotlib.pyplot as plt
        # we'll fit |S21|^2 to a lorentzian
        ampsqr = (np.absolute(self.z_data_raw[self._fid]))**2
        
        # make some estimates, resonant frequency should be the largest magnitude
        f0_ind = np.argmax(ampsqr)
        f0_est = self.f_data[f0_ind]
        # bandwidth is at half max, but we're looking at squared data, so look at 1/4 max
        A_est = np.absolute(self.z_data_raw[f0_ind])
        bw_id = np.argwhere(ampsqr > (A_est**2)/4)
        resbw = self.f_data[bw_id]
        #plt.plot(resbw,ampsqr[bw_id])
        #plt.show()
        #print(resbw[-1])
        #print(resbw[-1]-resbw[0])
        fwhm_est = (resbw[-1]-resbw[0])[0]
        #print('estimate fwhm {}'.format(fwhm_est))
        Q_est = 2*np.pi*f0_est/fwhm_est
        #print('estimates f0 {},Q {}, A {}'.format(f0_est,Q_est,A_est))
        
        # let's try to remove the delay using only 5 fwhm from estimated resonance
        fit_id = np.logical_and(self.f_data > (f0_est - 5*fwhm_est),self.f_data < (f0_est + 5*fwhm_est))
        self.delay = self._fit_delay(self.f_data[fit_id],self.z_data_raw[fit_id],0,maxiter=200)
        #print('delay: {:.6f} ns'.format(self.delay*1e9))
        z_data = self.z_data_raw*np.exp(2j*np.pi*self.delay*self.f_data)
        #plt.plot(self.z_data_raw[fit_id].real,self.z_data_raw[fit_id].imag,z_data.real,z_data.imag)
        #plt.show()
        xc,yc,r0 = self._fit_circle(z_data[fit_id])
        
        # center data for phase fit
        z_data_center = self._center(z_data,np.complex(xc,yc))
        
        
        # get Q and f0 from phase fit          
        theta, Q, f0 = self._phase_fit(self.f_data[self._fid],z_data_center[self._fid],0,Q_est,f0_est)
        #print('from phase fit, f0: {:.1f}, Q: {:.1f}'.format(f0,Q))
        
        # let's make the 'calibrated data' just have electrical delay removed, and rotated such that resonance
        # lies on real axis. It's important to note that we cannot do anything to normalize the magnitude. There
        # is no reference which we can normalize to.
        self.alpha = np.angle(np.complex(xc+r0*np.cos(theta),yc+r0*np.sin(theta)))
        self.z_data = z_data*np.exp(-1j*self.alpha)
        
        
        #lastly, let's dial in the amplitude properly
        A_est = 2*r0/Q
        fwhm = 2*np.pi*f0/Q
        fit_id2 = np.logical_and(self.f_data > (f0 - fwhm),self.f_data < (f0 + fwhm))
        ampsqr = (np.absolute(self.z_data_raw[fit_id2]))**2
        p = [f0,Q,A_est]
        popt, pcov = spopt.curve_fit(self._S21_sqr, self.f_data[fit_id2], ampsqr,p,
                                    bounds=([0.9*f0,0.8*Q,0.5*A_est],
                                           [1.1*f0,1.2*Q,2*A_est]))
        
        
#        self.fit_found = True
#        self.amp_trans = A
#        self.kappa = 2*np.pi*self.f0/self.Q
#        self.ringdown_time = (np.log(2)/np.pi) * self.Q / self.f0
        
        errors = np.sqrt(np.diag(pcov))
        fit_is_good = (errors/popt) < 0.5
        if all(fit_is_good):
            self.fit_found = True
            self.f0,self.Q,self.amp_trans = popt
            self.kappa = 2*np.pi*self.f0/self.Q
            self.ringdown_time = (np.log(2)/np.pi) * self.Q / self.f0
        else:
            print(p)
            print(popt)
            print(errors/popt)
            
    def get_single_photon_limit(self,unit='dBm'):
        '''
        returns the amout of power in units of W necessary
        to maintain one photon on average in the cavity
        unit can be 'dBm' or 'watt'
        '''
        if self.fit_found:
            k_c = 2*np.pi*self.f0/self.Qc
            k_i = 2*np.pi*self.f0/self.Qi
            Pwatt = (2.*np.pi*hbar*self.f0*(k_c+k_i)**2)/(4*k_c) # approx. hbar omega kappa /4 when very overcoupled
            if unit=='dBm':
                return Watt2dBm(Pwatt)
            elif unit=='watt':
                return Pwatt				  
        else:
            warnings.warn('Please perform the fit first',UserWarning)
            return None
        
    def get_snr(self):
        if not self.fit_found:
            return np.nan
        noise = np.std(np.absolute(self.z_data_sim - self.z_data_raw),dtype=np.float64)
        snr = (self.amplitude/noise)**2 if self.port_type is not 'T' else (self.amp_trans*self.Q/noise)**2
        return snr
    
    # adding a string method
    def __str__(self):
        if not self.fit_found:
            return 'Data has not been fitted'
        if self.port_type is not 'T':
            string = ('Frequency: {:.4f} GHz'.format(self.f0*1e-9)
                      + '\nTotal Q: {:d}'.format(int(np.round(self.Q)))
                      + '\nInternal Q: {:d}'.format(int(np.round(self.Qi)))
                      + '\nCoupling Q: {:d}'.format(int(np.round(self.Qc)))
                      + '\nFWHM: {:.5f} MHz'.format(self.kappa*1e-6/(2*np.pi))
                      + '\nKappa: {:.5f} MHz'.format(self.kappa*1e-6)
                      + '\nSingle Photon Power: {:.1f} dBm'.format(self.P1photon)
                      + '\nRingdown Time: {:.3f} us'.format(self.ringdown_time*1e6)
                      + '\nImpedance Mismatch {:.3f} degrees'.format(self.phi0*180/np.pi)
                      + '\nElectrical Delay: {:.6f} ns'.format(self.delay*1e9)
                      + '\nSNR: {:.1f}'.format(self.snr)
                     )
        else:
            string = ('Frequency: {:.4f} GHz'.format(self.f0*1e-9)
                      + '\nTotal Q: {:d}'.format(int(np.round(self.Q)))
                      + '\nFWHM: {:.5f} MHz'.format(self.kappa*1e-6/(2*np.pi))
                      + '\nKappa: {:.5f} MHz'.format(self.kappa*1e-6)
                      + '\nRingdown Time: {:.3f} us'.format(self.ringdown_time*1e6)
                      + '\nElectrical Delay: {:.6f} ns'.format(self.delay*1e9)
                      + '\nSNR: {:.1f}'.format(self.snr)
                     )
        return string
    
    def _S11_directrefl(self,f,fr=10e9,Ql=900,Qc=1000.,a=1.,alpha=0.,delay=.0):
        '''
        full model for direct reflection type resonances
        '''
        return a*np.exp(np.complex(0,alpha))*np.exp(-2j*np.pi*f*delay) * ( 2.*Ql/Qc - 1. + 2j*Ql*(fr-f)/fr ) / ( 1. - 2j*Ql*(fr-f)/fr )
    
    def _S21_notch(self,f,fr=10e9,Ql=900,Qc=1000.,phi=0.,a=1.,alpha=0.,delay=.0):
        '''
        full model for notch type resonances
        '''
        return a*np.exp(np.complex(0,alpha))*np.exp(-2j*np.pi*f*delay)*(1.-Ql/Qc*np.exp(1j*phi)/(1.+2j*Ql*(f-fr)/fr))
    
    def _S21_sqr(self,f,fr=10e9,Ql=1000,A=1):
        '''
        lorentzian model for direct transmission type resonances. note that |S21|^2 is lorentzian.
        '''
        return A**2/((1./(Ql**2)) + 4.*(((f/fr) - 1)**2))
    
    def _S21_direct(self,f,fr=10e9,Ql=1000,A=1,alpha=0,delay=0):
        '''
        full model for direct transmission type resonances.
        '''
        return A*Ql*np.exp(1j*(alpha-2*np.pi*f*delay))*(1 + 2j*Ql*(1-f/fr))/(1 + 4*(Ql**2)*((1-f/fr)**2))

    def add_calibration_data(self,f_data,z_data):
        self.calibration_data = {'f':np.array(f_data),'z':np.array(z_data)}
