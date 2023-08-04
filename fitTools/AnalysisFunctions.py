import numpy as np
#import peakutils
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        Basic functions for fitting
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def RB_1qb(m,p,A,B):
    return A*p**m+B

def StraightLine(x,a,b):
    return a*x+b

def T1_fit(x,Gamma,A,c):
    return A*np.exp(-x*Gamma)+c


def T1_fit_2(x,T1,A,c):
    return A*np.exp(-x/T1)+c

"""
def Ramsey_fit(x, T1_2, period, phase, T_phi,Amp,offset):
    return Amp*np.cos(2*np.pi*x/period+phase)*np.exp(-(x/T_phi)**2)*np.exp(-x/T1_2) + offset
"""

def fit_Ramsey(x_data, a, ramsey_frequency, t2, phase, DC_offset):
    return (a*np.exp(-x_data/t2)*np.cos(2*np.pi*(ramsey_frequency)*x_data + phase) + DC_offset)
"""
def fit_Ramsey(x_data, a, ramsey_frequency, t2, phase, DC_offset):
    return (a*np.exp(-x_data/t2)*np.cos(2*np.pi*(ramsey_frequency)*x_data + 0) + DC_offset)
"""
def fit_Ramsey_exp(x_data, a, ramsey_frequency, t2, b,t2_zero,phase, DC_offset):
    return (a*np.exp(-x_data/t2)*np.cos(2*np.pi*(ramsey_frequency)*x_data + phase/180*np.pi) +b*np.exp(-x_data/t2_zero)+ DC_offset)

def Rabi_fit(x, tau, period, phase, Amp, offset):
    return Amp*np.cos(2*np.pi*x/period+phase)*np.exp(-(x/tau)) + offset

def ExpGaussDecay(x,T1_2,T_phi,Amp,offset):
    return Amp*np.exp(-x/T1_2)*np.exp(-(x/T_phi)**2)+offset

def Qfreq(V,offset,phi0,A):
    return A*np.sqrt(np.absolute(np.cos(np.pi*(V-offset)/phi0)))

def dQfdV(V,offset,phi0,A):
    return -np.sign(np.cos(np.pi*(V-offset)/phi0))*A*np.pi*np.sin(np.pi*(V-offset)/phi0)/(2*phi0*np.sqrt(np.absolute(np.cos(np.pi*(V-offset)/phi0))))

def Lorentzian(x,amp,width,center,offset):
    return amp*width/((x-center)**2+width**2)+offset

def cosFit(x,A,T,phase,c): # Thorvald name for a cosine function
    y = A*np.cos(2*np.pi*x/T + phase) + c
    return y

def Echo_fit(x,Amp,T2,offset):
    return Amp*np.exp(-x/T2) + offset

#use FFT to give a guess on rabi peroid
def fft_period(ydata,xdata):
   """FFT of ydata, return signal period in ns
   d: sample distance, 1/sampling rate
   n: optional, length of data by defalut
   test with: fft_period(ydata,xdata,n=300)
   """
   n = len(xdata)
   d =(xdata[1]-xdata[0]) # unit: us
   fydata = np.abs(np.fft.fft(ydata,n))
   freq = np.fft.fftfreq(n,d)
   guess_freq = freq[np.argmax(fydata)]
   plt.plot(freq,fydata)
   guess_period = 1/guess_freq
   print("The guess period is {} us".format(guess_period))
   return guess_period

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        Log inspecter and ancillay functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def LogInspect(exp):
    """
    Basic function that prints an 'overview' of the contents of a logfile


    """
    print('>STEP CHANNELS')
    step_chans = exp.getStepChannels()
    for channel in step_chans:
        if (len(channel['values']) > 1):
            print('{1:30} ({2:^3}) {0:>4} '.format(len(channel['values']), channel['name'], channel['unit']))

    print('>LOG CHANNELS')
    log_channels = exp.getLogChannels()
    for channel in log_channels:
         print('{0:30} ({1:^3})'.format(channel['name'],channel['unit']))


## Following function copied from Labber to ensure things are consistent.
def calcRotationAngle(vComplex):
    """Calculate angle that will rotate the signal in the input complex vector
    to the real component"""
    # remove nan's and infinites
    vComplex = vComplex[np.isfinite(vComplex)]
    # make sure we have data, check if complex
    if len(vComplex)<2:
        return 0.0
    if not np.any(np.iscomplex(vComplex)):
        return 0.0
    vPoly1 = np.polyfit(vComplex.real, vComplex.imag, 1)
    vPoly2 = np.polyfit(vComplex.imag, vComplex.real, 1)
    # get slope from smallest value of dy/dx and dx/dy
    if abs(vPoly1[0]) < abs(vPoly2[0]):
        angle = np.arctan(vPoly1[0])
    else:
        angle = np.pi/2.0 - np.arctan(vPoly2[0])
        if angle > np.pi:
            angle -= np.pi
    angle = -angle
    # try to make features appear as peaks instead of dips by adding pi
    data = np.real(vComplex * np.exp(1j*angle))
    meanValue = np.mean(data)
    # get metrics
    first = abs(data[0] - meanValue)
    low = abs(np.min(data) - meanValue)
    high = abs(np.max(data) - meanValue)
    # method: use first point if first signal > 0.5 of max or min
    if first > 0.5*max(low,high):
        # approach 1: check first point (good for oscillations)
        if data[0] < meanValue:
            angle += np.pi
    else:
        # approach 2: check max/min points
        if high < low:
            angle += np.pi
#    # approach 2: check mean vs median (good for peaks)
#    if meanValue < np.median(data):
#        angle += np.pi
    return angle


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        Functions for spectroscopy peaks
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#def specPeakFinder(dataX,dataY,data_Thresh,PF_Thresh,PF_Dist,SG_Window,SG_Poly):
#    '''
#    dataX = x-axis datapoints
#    dataY = yaxis datapoints
#    data_Thresh = threshold value to use when figuring out if there's even a peak in the data:
#        |max(y1) - <y1>|/std(y1) > datapeakThresh
#    PF_thresh = threshold value for peakfinder between 0 and 1. 1 is max height of data
#    PF_dist = minimum distance between neighboring peaks
#    SG_window = size of window used for Savitzky-Golay filtering
#    SG_poly = polynomial to used for the SG filtering
#    '''
#    angle = calcRotationAngle(dataY) #find angle in radians to rotate data by
#    y = np.real(np.exp(1j*angle)*dataY)
#
#    ymin  = np.min(y)
#    ymax  = np.max(y)
#
#    y_smooth = savgol_filter(y,SG_Window,SG_Poly) # smooth
#
#
#    # Ami thinks this is wrong 12/9/17
#    #maxPeakHeight = np.abs(np.abs(ymin) - np.abs(ymax))
#    maxPeakHeight = np.abs(ymax - ymin)
#
#    if maxPeakHeight/np.std(y) > data_Thresh: # determine if a reasonable peak even exists
#        # Ami debugging 12/9/17
#        #y_smooth = savgol_filter(y,SG_Window,SG_Poly) # smooth
#        peakindexes = peakutils.indexes(y_smooth, thres=PF_Thresh, min_dist=PF_Dist)
#
#        if len(peakindexes) > 0:
#            xvalpeak = dataX[peakindexes[-1]]
#            yvalpeak = y[peakindexes[-1]]
#            peakindex = peakindexes[-1]
#
#            return peakindex, xvalpeak, yvalpeak
#    else: # just make sure we handle a no-peak properly
#        pass



def getMultiQBspec(dataFile, qubitDict,data_Thresh,PF_Thresh,PF_Dist,SG_Window,SG_Poly):

    datapoints = dataFile.getNumberOfEntries()
    frequencyChannelName = dataFile.getStepChannels()[0]['name']
    fluxVals = [dataFile.getEntry(j)['Bobbin - Voltage'][0] for j in range(datapoints)]

    # defining constants
    numAnalyzedQubits = len(qubitDict.keys())


    #initializing empty lists
    # analysis results will be indexed by [qubitIndex][pointNum]
    # begin by making a list of empty lists. Each inner list will hold the data for one qubit
    peakIndexes = [[] for i in range(numAnalyzedQubits)]
    xpeakLists = [[] for i in range(numAnalyzedQubits)]
    ypeakLists = [[] for i in range(numAnalyzedQubits)]
    fluxValLists = [[] for i in range(numAnalyzedQubits)]


    # Going through the data file for each qubit, and extracting the bobbin voltage at which that qubit is excited
    for qubitIndex in qubitDict:
        currentQubit = qubitDict[qubitIndex]
        print('qubit {}'.format(currentQubit))

        for j in range(datapoints):
            # extracting data from .hd5 file
            data = dataFile.getEntry(j)
            x = data[frequencyChannelName]
            y = data['MQ PulseGen - Voltage, {}'.format(currentQubit)]

            # peakfinder automatically does rotation
            peaksFound = specPeakFinder(x,y,data_Thresh,PF_Thresh,PF_Dist,SG_Window,SG_Poly)
            try:
                    peakIndexes[qubitIndex].append(peaksFound[0])
                    xpeakLists[qubitIndex].append(peaksFound[1]/1E9) # return values in GHz
                    ypeakLists[qubitIndex].append(peaksFound[2])
                    fluxValLists[qubitIndex].append(fluxVals[j])
            except TypeError:
                #print('{} at {:2.3f} V: No suitable peaks found'.format(currentQubit,fluxVals[j]))
                pass

    return (peakIndexes, xpeakLists, ypeakLists, fluxValLists)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        Functions for noise data
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def serial_corr(A,lag=1):
    n = len(A)
    corr = np.corrcoef(A[lag:],A[:n-lag])[1,0]
    return corr
    
def autocorr(A,sampleRate=1):
    lags = range(len(A)//2)
    corrs = [serial_corr(A, lag) for lag in lags]
    time = [lag/sampleRate for lag in lags]
    return time, corrs


def extractPeakInformation(dataFile, dataType, qubitList, angleDict, pointStart, pointEnd, widthGuess, widthErr = .01):
    """
    Find peak locations from traces of hd5 file.
    Returns dictionaries containing a list of trace centers, fit errors and x values of successful fits for each qubit.
    (ie, returns [centerListDict, errorListDict, xPointsListDict]), where centerListsDict['QB1'] is a list of center locations
    dataType: string describing Labber parameter varied for x-axis of fit (ex/ 'Common Drive - Frequency', 'Bobbin - Voltage'
    dataFile: Labber.LogFile object formed from .hd5 file to be analyzed
    qubitList: list naming qubits to be analyzed (ex/ ['QB1', 'QB5']
    angleDict: dictionary containing angles by which to rotate data for each qubit (ex/ {'QB1': 0, 'QB5': 180})
    pointStart: first trace in file to be analyzed
    pointEnd: last trace in file to be analyzed
    widthGuess: Float. Guess for peak width in units of dataType given
    widthError: Float used to determine maximum acceptable error in a fit. Percent error of widthGuess. (ex/ 0.01 for 1% error)
    """
    # initializing empty lists
    # analysis results will be indexed by [qubitIndex][pointNum]
    # begin by making a dictionary of empty lists. Each dictionary entry will hold the data for one qubit
    centerListDict = {}
    errorListDict = {}
    xPointsListDict = {}
    widthListDict = {}

    for qubitName in qubitList:
        centerListDict[qubitName] = []
        errorListDict[qubitName] = []
        xPointsListDict[qubitName] = []
        widthListDict[qubitName] = []


    # Going through the data file for each qubit, and extracting the bobbin voltage at which that qubit is excited
    for currentQubit in qubitList:
        print('qubit {}'.format(currentQubit))

        # for each time slice
        for pointNum in range(pointStart, pointEnd):

            # extracting data from .hd5 file
            data = dataFile.getEntry(pointNum)
            x = data[dataType]
            y = np.real(np.exp(1j*angleDict[currentQubit]*np.pi/180)*data['MQ PulseGen - Voltage, {}'.format(currentQubit)])

            # try fitting the data to a Lorentzian
            AmplGuess = np.abs(np.abs(np.max(y)) - np.abs(np.min(y)))
            CenterGuess = x[np.argmax(y)]
            OffsetGuess = np.mean(y)
            try:
                p0 = [AmplGuess,widthGuess,CenterGuess,OffsetGuess] # [amplitude,width,center,offset]
                popt, pcov = curve_fit(Lorentzian,x,y,p0)
                perr = np.sqrt(np.diag(pcov))

                # if the fit has a width that roughly corresponds to what we expect
                # and an amplitude that is more than twice the std of the sample, consider the fit good, and use it
                if perr[2] < widthErr*widthGuess and popt[0]> 2*np.std(y):#perr[2] < 20*1E-6 and popt[0]> 1*np.std(y):
                    centerListDict[currentQubit].append(popt[2])
                    errorListDict[currentQubit].append(perr[2])
                    widthListDict[currentQubit].append(popt[1])
                    xPointsListDict[currentQubit].append(pointNum)

            # announce failed fits
            except RuntimeError:
                print('Fail fit at {:d} for {}'.format(pointNum, currentQubit))
                continue
    return (centerListDict, errorListDict, xPointsListDict)



def plotNoiseData(xPointsListsDict, normalizedCenterListsDict, errorListsDict, stdDict, pointStart, pointEnd, titleText, yLabel, unitName, scalingFactor):
    """
    Plotting the normalized traces
    xPointsListsDict, normalizedCenterListsDict, errorListDict: returned from getNormalizedCenterDict
    stdDict: returned from getStdDict. Used in legend
    pointStart: first trace number to be plotted
    pointEnd: last trace number to be plotted
    titleTest: String containing desired title for plot
    yLabel: String contining desired label for y axis
    unitName: string containing desired unit for the std in the legend
    scalingFactor: float used to scale data to desired units, if needed. For example, normalizedCenterListsDict might be in Hz, but one
         might want to plot the data in kHz, which would lead to a scaling factor of 1e-3.
    """
    fig,ax = plt.subplots(1,1,figsize=(14,10))

    for qubitName in xPointsListsDict.keys():

        # for each qubit, define parameters to be plotted
        xPoints = xPointsListsDict[qubitName]
        yPoints = np.multiply(normalizedCenterListsDict[qubitName], scalingFactor)
        yErrors = np.multiply(errorListsDict[qubitName], scalingFactor)
        currentLabel = '{}: Std {:1.2f} {}'.format(qubitName, float(stdDict[qubitName]*scalingFactor), unitName)

        ax.errorbar(xPoints,yPoints,yerr = yErrors,fmt='.-',ecolor='k',markersize='5',linewidth='1',label=currentLabel)

    ax.set_xlim([pointStart,pointEnd])
    ax.set_ylabel(yLabel)
    ax.set_xlabel('Repetition (\#)')
    ax.set_title(titleText)
    #debugging
    ax.legend(loc=0)

def getNormalizedCenterDict(centerListDict):
    """
    Given list of lists of peak locations, subtract the mean from each list.
    centerListDict: Returned from extractPeakInformation
    """
    qubitList = centerListDict.keys()
    normalizedCenterListDict = {}

    # subtract the mean from each qubit's voltage, to better compare deviations
    for qubitName in qubitList:
        normalizedCenterListDict[qubitName] = centerListDict[qubitName] - np.mean(centerListDict[qubitName])

    return normalizedCenterListDict


def getStdDict(normalizedCenterListDict):
    """
    Given list of lists of peak locations, find the standard deviation of each list
    normalizedCenterListDict: Returned from extractPeakInformation
    """
    qubitList = normalizedCenterListDict.keys()
    stdDict = {}

    # get std for each qubit
    for qubitName in qubitList:
        stdDict[qubitName] = np.std(normalizedCenterListDict[qubitName])
    return stdDict

def plotSingleFit(dataFile, dataType, qubitName, pointNum, angleDict, widthGuess, widthErr = .01):
    """
    If extractPeakInformation isn't working well, use this function to plot a single trace and its fit of a single point
    dataFile: Labber.LogFile object formed from .hd5 file to be analyzed
    dataType: string describing Labber parameter varied for x-axis of fit (ex/ 'Common Drive - Frequency', 'Bobbin - Voltage'
    qubitName: String naming qubits to be analyzed (ex/ 'QB1')
    pointNum: Integer, trace number to be used
    angleDict: dictionary containing angles by which to rotate data for each qubit (ex/ {'QB1': 0, 'QB5': 180})
    widthGuess: Float. Guess for peak width in units of dataType given
    widthError: Float used to determine maximum acceptable error in a fit. Percent error of widthGuess. (ex/ 0.01 for 1% error)
    """

    data = dataFile.getEntry(pointNum)
    x = data[dataType]
    y = np.real(np.exp(1j*angleDict[qubitName]*np.pi/180)*data['MQ PulseGen - Voltage, {}'.format(qubitName)])

    # try fitting the data to a Lorentzian
    AmplGuess = np.abs(np.abs(np.max(y)) - np.abs(np.min(y)))
    CenterGuess = x[np.argmax(y)]
    OffsetGuess = np.mean(y)
    try:
        p0 = [AmplGuess,widthGuess,CenterGuess,OffsetGuess] # [amplitude,width,center,offset]
        popt, pcov = curve_fit(Lorentzian,x,y,p0)
        perr = np.sqrt(np.diag(pcov))

        # if the fit has a width that roughly corresponds to what we expect
        # and an amplitude that is more than twice the std of the sample, consider the fit good, and use it
        if perr[2] < widthErr*widthGuess and popt[0]> 2*np.std(y):
            print('fit succeeded')
        else:
            print('fit did not pass standards')
            print('perr[2] is {} and should be less than {}'.format(perr[2], widthErr*widthGuess))
            print('popt[0] is {} and should be greater than {}'.format(popt[0], 2*np.std(y)))

    # announce failed fits
    except RuntimeError:
        print('Fail fit at {:d} for {}'.format(pointNum, currentQubit))


    plt.plot(x,y)
    y_fit = [Lorentzian(xi, popt[0], popt[1], popt[2], popt[3]) for xi in x]
    plt.plot(x,y_fit)
    plt.show()

def findCorrelations(xPointsListDict, normalizedCenterListDict, pointStart, pointEnd):
    """
    Given fitted peaks of data, return correlation matrix
    xPointsListDict: returned from extractPeakInformation. A dictionary which, for each qubit, contains a list of traces for which fitting
        was successful
    nomrmalizedCenterListDict: returned from extractPeakInformation. A dictionary which, for each qubit, contains a list of the center
        values of the fitted peaks.
    pointStart, pointEnd: integers. Traces from pointStart to pointEnd will be analyzed. Must be the same as those used to generate
        xPointsListDict and normalizedCenterListDict.
    """
    # If we had fits for each qubit for all traces, we could just call
    # >> dataCorr = [normalizedCenterListDict[qubitName] for qubitName in qubitList]
    # >> return np.corrcoef(dataCorr)
    # However, we don't have a fit for all qubits for all traces. Most of this function is finding the traces for which all qubits
    # have good fits, and keeping track of those points so we can use them for np.corrcoef. This means searching through xPointsListDict
    # to find values which are in each entry.

    # We could do this with a structure like:
    # >> for point in range(pointStart, pointEnd)
    # >>>>> pointInAllLists = True
    # >>>>> for qubitName in qubitList
    # >>>>>>>> if not(point in xPointsListDict[qubitName]):
    # >>>>>>>>>>> pointInAllLists = False
    # But that would involve looping through long lists many times as we check if each point is in each list. Since xPointsListDict entries
    # entries are sorted, we can do better than that.

    # Instead, we loop through each entry of xPointsListDict (ie through each list of traces used for a particular qubit) once.
    # As we look through each list for points, we keep track of which index we have checked up to, in indexTrackingDict.
    # For each point and each list of traces xPointsList[qubitName], the point is either at the next index of the list, or it is not
    # included in the list. Each time we find a point, we increment the tracker index.

    # For each point that all qubits have in common, we want to find the center value for all qubits. We create a list of these for each
    # qubit, and store them in normalizedCentersSharedXPointsDict.

    # initializing lists of shared points to be empty
    qubitList = xPointsListDict.keys()
    normalizedCentersSharedXPointsDict = {}
    indexTrackingDict = {}

    # More initialization
    # We set our starting search index for each list of traces to be 0.
    # We also set each entry in normalizedCentersSharedXPointsDict to be an empty list, to be filled when we find traces for which all
    # qubits have good fits.
    for qubitName in qubitList:
        indexTrackingDict[qubitName] = 0
        normalizedCentersSharedXPointsDict[qubitName] = []

    # looping through all of the traces that the qubits could share
    for point in range(pointStart, pointEnd):
        pointInAllLists = True

        # for each qubit, check to see if there was a good fit at this point
        for qubitName in qubitList:

            # if this trace is in the 'we found good fits here' list for this qubit, it will be at this index
            # we know this because the list is sorted, and we are looking through it in ascending order.
            qubitIndex = indexTrackingDict[qubitName]

            # check to see if the current trace is in the  the 'we found good fits here' list for this qubit
            # the first clause keeps us from looking at entries past the end of the list
            if (qubitIndex < len(normalizedCenterListDict[qubitName])) and xPointsListDict[qubitName][qubitIndex] == point:

                # if we found the trace in this list, increment the 'tracked index' so we look at the next entry when we are trying to
                # see if the next trace is present

                indexTrackingDict[qubitName] +=1

            # if we don't find the trace in this list of 'we found good fits for this qubit at these traces', it isn't in all lists.
            else:
                pointInAllLists = False

        # if we do have good fits at this trace for all qubits, then jot down the center values so we can see if they are correlated
        if pointInAllLists:

            for qubitName in qubitList:
                 # some book-keeping here; because we incremented the index above, we have to decrement it to get the correct value
                foundIndex = indexTrackingDict[qubitName] -1
                normalizedCentersSharedXPointsDict[qubitName].append(normalizedCenterListDict[qubitName][foundIndex])

    # and now that we have our lists of data, finding the correlations is very easy
    dataCorr = [normalizedCentersSharedXPointsDict[qubitName] for qubitName in qubitList]
    return np.corrcoef(dataCorr)



def findCloseValue(dataArray, value):
    '''
    dataArray: a sorted 1d array (low value to high value)
    value: a float
    returns index of array with an element adjacent to where 'value' would be in the list. Similar to a binary search
    '''

    startIndex = 0
    stopIndex = len(dataArray) - 1

    # if value is lower or higher than any element in the list, return first or last index, respectively
    if value < dataArray[startIndex]:
        return startIndex
    if value > dataArray[stopIndex]:
        return stopIndex

    while (stopIndex - startIndex) > 1:
        midIndex = int((startIndex + stopIndex)/2)

        if dataArray[midIndex] == value:
            return midIndex

        elif dataArray[midIndex] > value:
            stopIndex = midIndex

        else:
            startIndex = midIndex

    startDistance =  dataArray[startIndex] - value
    stopDistance = dataArray[stopIndex] - value

    if startDistance < stopDistance:
        return startIndex

    else:
        return stopIndex

            
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        Functions for transition width analysis
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def prob_func(phi_z, z0, w):
    """Describes probability S-curve"""
    return (1/2*(1+np.tanh((phi_z-z0)/w)))


def calculateProbabilyCurves(data = None, x_data = None, threshold = None):
    """
    data should a dict from 
    be a 2D array, with n_shots along  one dim, and the swept flux on the other
    
    threshold is the value used to separate 0 and 1. Can be complex, and will use the distance in complex plane
    to determine if a 0 or a 1
    
    Fits the data to prob_func and returns fit parameters as a list
    """
    if type(data) is dict:
        vals = data['values']
        x_data = data['x'][0]
        
    else:
        vals = data
     
    if vals[0,:].size == x_data.size: 
        vals = vals.T

    if threshold is None:
        threshold = vals.mean()
    
    if np.iscomplex(threshold):
        # Rotate so magnitude of the threshold is along the real axis 
        # Do a positive rotation
        rotate_angle = np.exp(-1j*np.arctan2(np.imag(threshold),np.real(threshold)))
        vals = vals*rotate_angle
        # after rotation the blobs should have the same real value and only differ along the imaginary
        # This is a 0 if it is less than 0 and a 1 if greater than
        pr_array = [(tr < 0.0).sum()/len(tr) for tr in np.imag(vals)]  # one liner that bins as above for the whole dataset

    else:
        vals = np.abs(vals)
        pr_array = [(tr > threshold).sum()/len(tr) for tr in vals]  # one liner that bins as above for the whole dataset
        # The data is often offset from zero. We need to find the closest value to zero and move the data for fitting:
    # Is the data rising l->r or r->l?
    if np.array(pr_array[0:3]).mean() < np.array(pr_array[-3:]).mean(): 
        rise = True
    else:
        rise = False
    
    
    if np.min(pr_array)<0.5:

        if rise:
            i = 0
            while (pr_array[i] < 0.5):
                i += 1
                x_adjusted = x_data - x_data[i]
                x_offset = x_data[i]
            # fit positive data:
            else:
                x_adjusted = x_data
        
 
        else: # fit negative sloping data   
            i = 0
            while (pr_array[i] > 0.5):
                i += 1
                x_adjusted = x_data - x_data[i]
                x_offset = x_data[i]
                pr_array = 1- pr_array
        
    else:
        x_adjusted = x_data
        x_offset = 0.0
    
    pars2, covar2 = curve_fit(prob_func, x_adjusted, pr_array, [0, 0.002])
    z0 = pars2[0]
    w = pars2[1]
    z0_err = np.diag(covar2)[0]
    w_err = np.diag(covar2)[1]

    return (pr_array, z0, w, z0_err , w_err, x_offset)
    
    

def calculateProbabilyCurves_2D(data = None, threshold = None):
    """
    Takes a data dictionary of flux vs shots vs. some other dimension
    
    Returns: array of Probabilities and fit values
    """
    
    # do the fitting for each step of the third dimension
#    pr_array, z0, w, z0_err , w_err, x_offset = [],[],[],[],[],[] # init empty arrays to return later
    results = ([],[],[],[],[],[])
    
    for n, x_3 in enumerate(data['x'][2]):
       fit_results =  calculateProbabilyCurves(data = data['values'][:,:,n], x_data= data['x'][0], threshold= threshold)
       for ii, r in enumerate(fit_results):
           results[ii].append(r)
    # rearrange results into indvidual lists
    return results