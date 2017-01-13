#
#
# utilities.py
# created winter 2016 g.c.rich
#
# general purpose functions for MCMC tof fitting project
#
# TODO: add effective posterior predictive check (PPC)


import numpy as np
from numpy import inf
from scipy.integrate import quad
from scipy.interpolate import interp1d
import scipy.optimize as optimize
from scipy.special import erf
import csv as csvlib
from constants.constants import (masses, qValues, distances, physics)



##############
# vars for binning of TOF 
tof_nBins = 25
tof_minRange = 180.0
tof_maxRange = 205.0
tof_range = (tof_minRange,tof_maxRange)


# PARAMETER BOUNDARIES
min_e0, max_e0 = 800,1100
min_e1,max_e1= -150, 0
min_e2,max_e2 = -30, 0
min_e3,max_e3 = -10, 0
min_sigma,max_sigma = 40, 100


def normalizeVec(a):
    """
    Normalize a 1D vector
    Simple, maybe inefficient
    Doesn't check to make sure that you're supplying a valid vector!
    """
    integrated = np.sum(a)
    return a/integrated


def getDDneutronEnergy(deuteronEnergy, labAngle = 0):
    """
    Get the energy of neutrons produced by DDN reaction
    Function accepts the deuteron energy (in keV) and the angle (in lab\
    frame of reference) at which the neutron is emitted.
    Returns neutron energy in keV
    """     
    neutronAngle_radians = labAngle * np.pi / 180
    rVal = np.sqrt(masses.deuteron * masses.neutron*deuteronEnergy) / \
                   (masses.neutron + masses.he3) * \
                   np.cos(neutronAngle_radians)
    sVal = (deuteronEnergy *( masses.he3 - masses.deuteron) +
            qValues.ddn * masses.he3) / (masses.neutron + masses.he3)
    sqrtNeutronEnergy = rVal + np.sqrt(np.power(rVal,2) + sVal)
    return np.power(sqrtNeutronEnergy, 2)
    
def getTOF(mass, energy, distance):
    """
    Compute time of flight, in nanoseconds, given\
    mass of particle (in keV/c^2), the particle's energy (in keV),\
    and the distance traveled (in cm).
    Though simple enough to write inline, this will be used often.
    """
    velocity = physics.speedOfLight * np.sqrt(2 * energy / mass)
    tof = distance / velocity
    return tof
    
    
def generateModelData(params, standoffDistance, nSamples):
    """
    Generate some fake data from our mode with num samples = nSamples
    params is an array of the parameters, [e0, e1, sigma]
    Returns a tuple of len(nSamples), [x, ed, en, tof]
    """
    initialEnergy, eLoss, e2, e3, sigma = params
    
    data_x=np.random.uniform(low=0.0, high=distances.tunlSSA_CsI.cellLength, size=nSamples)
    meanEnergy = initialEnergy + eLoss*data_x + \
                              e2*np.power(data_x,2) + e3 * np.power(data_x,3)
    data_ed= np.random.normal(loc=meanEnergy, scale=sigma)
    data_en = getDDneutronEnergy(data_ed)
    
    neutronDistance = standoffDistance + (distances.tunlSSA_CsI.cellLength - data_x) + \
                        distances.tunlSSA_CsI.zeroDegLength/2
    neutronTOF = getTOF(masses.neutron, data_en, neutronDistance)
    effectiveDenergy = (initialEnergy + data_ed)/2
    deuteronTOF = getTOF( masses.deuteron, effectiveDenergy, data_x )
    data_tof = neutronTOF + deuteronTOF
    
    data = np.column_stack((data_x,data_ed,data_en,data_tof))
    return data
    
def lnlike(params, observables, nDraws=1000000):
    """
    Evaluate the log likelihood given a set of params and observables
    Observables is a vector; a histogrammed time distribution
    Params is a list of [initial D energy, E_D loss, sigma]
    nDraws is the number of points drawn from (energy,location) distribution\
    which are used to produce the PDF evaluations at different TOFs
    """
    evalData=generateModelData(params, distances.tunlSSA_CsI.standoffMid, nDraws)
    evalHist, evalBinEdges = np.histogram(evalData[:,3], tof_nBins, tof_range,
                                          density=True)
    logEvalHist = np.log(evalHist)
    #print(logEvalHist)
    # find what TOFs have zero observed data
    # we'll use this to handle cases where we might wind up with -inf*0
    # likelihood is fine if PDF is 0 somewhere where no data is found
    # without checks though, ln(PDF=0)=-inf, -inf*0 = nan
    # however, if PDF is 0 (lnPDF=-inf) where there IS data, lnL should be -inf
    zeroObservedIndices = np.where(observables == 0)[0]
    for idx in zeroObservedIndices:
        if logEvalHist[idx] == -inf:
            logEvalHist[zeroObservedIndices] = 0
    
    loglike = np.dot(logEvalHist,observables)
    return loglike
    
    

def lnprior(theta):
    """
    Evaluate the log priors for parameter list theta
    """
    e_0, e_1, e_2, e_3, sigma = theta
    if min_e0 < e_0 < max_e0 and min_e1 <e_1< max_e1 and min_e2 < e_2<max_e2 \
    and min_e3 < e_3 < max_e3 and min_sigma < sigma < max_sigma:
        return 0
    return -inf
    
def lnprob(theta, observables):
    """
    Evaluate the log probability
    theta is a list of the model parameters
    observables is, in this case, a histogram of TOF values
    """
    prior = lnprior(theta)
    if not np.isfinite(prior):
        return -inf
    return prior + lnlike(theta, observables)
   
    
def readMultiStandoffTOFdata(filename):
    """Read in data from TAC for multiple-standoff runs
    
    Specify filename to access.
    """
    lowerBinEdges =[]
    tofCounts=[]
    with open(filename,'r') as tofFile:
        csvreader = csvlib.DictReader(tofFile, delimiter='\t', 
                                  fieldnames=['lowEdge','run0',
                                  'run1','run2','run3'])
        for row in csvreader:
            lowerBinEdges.append(float(row['lowEdge']))
            tofCounts.append([float(row['run0']), float(row['run1']),
                                    float(row['run2']), float(row['run3'])])
    tofData = np.column_stack((lowerBinEdges,tofCounts))
    return tofData


class beamTimingShape:
    """Class with timing parameters for the gamma peak in TOF spectra
   
    Assumes that the peak is modeled with a gaussian convolved with an exponential tail
    The relevant parameters are then sigma and tau
    
    Note that right now these are tuned for TUNL CsI run at SSA
    In long run, this class should not be so specific
    
    for an exponential convolved with a gaussian, see
    https://ocw.mit.edu/courses/chemistry/5-33-advanced-chemical-experimentation-and-instrumentation-fall-2007/labs/laser_appendix3.pdf
    
    """
    def __init__(self):
        self.sigma = 1.2474 # ns, from roofit
        self.tau = 0.92503
        
        
        self.binnedTOF = True # flag to use binned values
        # presently there's not necessarily a plan to handle unbinned
        # but keep this flag here for forward compatibility
        self.tofBinWidth = 1 # width of bins in ns
    
        # set up the range that we'll use for the sliding window
        # right now they're just conservatively large
        # TODO: the treatment here may assume a bin width of 1 in the use of ceil
        self.windowRange_min = np.ceil(-1.0 * 5 * self.sigma)
        self.windowRange_max = np.ceil(10 * self.tau)
        self.window_nBins = self.windowRange_max - self.windowRange_min
    
        self.binCenters = np.linspace( (self.windowRange_min +
                                        self.tofBinWidth/2),
                                      (self.windowRange_max -
                                       self.tofBinWidth/2), self.window_nBins)
                                       
        # the idea is that this variable isnt persistent, may not be done correctly
        tempTiming = self.evaluateTimingDist(self.binCenters)
    
        # now we want to normalize the timing distribution
        # hopefully by doing this (once) we'll make it so that when we convolve
        # with the other features it does not UNnormalize the other data
        self.timingDistribution = normalizeVec(tempTiming)
        

    def evaluateTimingDist(self, time):
        """
        Get value for the timing distribution given a time from t0 in ns
        """
        expArg = self.sigma**2 / (2 * self.tau**2) - time / self.tau
        erfArg = ((self.sigma**2 - time * self.tau)/
                  (np.sqrt(2)*self.sigma*self.tau))
        timingDistVal = np.exp(expArg) * (1- erf(erfArg))
        return timingDistVal

    def applySpreading(self, tofDistribution):
        """Convolve the beam timing shape with raw TOF model data
        
        tofDistribution should be a numpy array
        binning needs to be accounted by user
        """
        return np.convolve(tofDistribution, self.timingDistribution, 'same')



class ddnXSinterpolator:
    """Class which handles spline interpolation of DDN cross section"""
    
    def __init__(self):
        # build the interpolation object
        # first have to populate the arrays of XS data
        self.dEnergies = []
        for e in range(20, 101, 10):
            self.dEnergies.append( float(e) )
        for e in range(150, 1001, 50):
            self.dEnergies.append( float(e) )
        for e in range(1100, 3001, 100):
            self.dEnergies.append( float(e) )
        for e in range(3500, 10001, 500):
            self.dEnergies.append( float(e) )
        
        self.ddnSigmaZero = []
        self.ddnSigmaZero.append( 0.025 )
        self.ddnSigmaZero.append( 0.125 )
        self.ddnSigmaZero.append( 0.31 )
        self.ddnSigmaZero.append( 0.52 )
        self.ddnSigmaZero.append( 0.78 )
        self.ddnSigmaZero.append( 1.06 )
        self.ddnSigmaZero.append( 1.35 )
        self.ddnSigmaZero.append( 1.66 )
        self.ddnSigmaZero.append( 2.00 )
        self.ddnSigmaZero.append( 3.33 )
        self.ddnSigmaZero.append( 4.6 )
        self.ddnSigmaZero.append( 5.9 )
        self.ddnSigmaZero.append( 7.1 )
        self.ddnSigmaZero.append( 8.3 )
        self.ddnSigmaZero.append( 9.4 )
        self.ddnSigmaZero.append( 10.4 )
        self.ddnSigmaZero.append( 11.4 )
        self.ddnSigmaZero.append( 12.4 )
        self.ddnSigmaZero.append( 13.4 )
        self.ddnSigmaZero.append( 14.3 )
        self.ddnSigmaZero.append( 15.1 )
        self.ddnSigmaZero.append( 15.8 )
        self.ddnSigmaZero.append( 16.5 )
        self.ddnSigmaZero.append( 17.2 )
        self.ddnSigmaZero.append( 17.8 )
        self.ddnSigmaZero.append( 18.4 )
        self.ddnSigmaZero.append( 19.0 )
        self.ddnSigmaZero.append( 20.0 )
        self.ddnSigmaZero.append( 21.0 )
        self.ddnSigmaZero.append( 21.9 )
        self.ddnSigmaZero.append( 22.7 )
        self.ddnSigmaZero.append( 23.4 )
        self.ddnSigmaZero.append( 24.0 )
        self.ddnSigmaZero.append( 24.6 )
        self.ddnSigmaZero.append( 25.2 )
        self.ddnSigmaZero.append( 25.8 )
        self.ddnSigmaZero.append( 26.4 )
        self.ddnSigmaZero.append( 26.9 )
        self.ddnSigmaZero.append( 27.5 )
        self.ddnSigmaZero.append( 28.0 )
        self.ddnSigmaZero.append( 28.4 )
        self.ddnSigmaZero.append( 28.9 )
        self.ddnSigmaZero.append( 29.3 )
        self.ddnSigmaZero.append( 29.8 )
        self.ddnSigmaZero.append( 30.3 )
        self.ddnSigmaZero.append( 30.7 )
        self.ddnSigmaZero.append( 31.2 )
        self.ddnSigmaZero.append( 33.5 )
        self.ddnSigmaZero.append( 35.7 )
        self.ddnSigmaZero.append( 37.8 )
        self.ddnSigmaZero.append( 40.0 )
        self.ddnSigmaZero.append( 41.5 )
        self.ddnSigmaZero.append( 42.9 )
        self.ddnSigmaZero.append( 43.8 )
        self.ddnSigmaZero.append( 44.6 )
        self.ddnSigmaZero.append( 45.2 )
        self.ddnSigmaZero.append( 45.7 )
        self.ddnSigmaZero.append( 46.1 )
        self.ddnSigmaZero.append( 46.4 )
        self.ddnSigmaZero.append( 46.5 )
        self.ddnSigmaZero.append( 46.5 )
        
        
        self.ddnXSfunc = interp1d(self.dEnergies, self.ddnSigmaZero,
                                  kind='cubic')
    
    def evaluate(self,deuteronEnergy):
        # check  to see if we have requested an energy outside of range
        # if so, return the XS value at the end of the range
        # this is not ideal, but we really ought not find these regions matter
        if len(deuteronEnergy)==1:
            if deuteronEnergy <= 20:
                return 0.025
            if deuteronEnergy >= 10000:
                return 46.5
        else:
            lowIndices = np.where(deuteronEnergy <= 20)
            highIndices = np.where(deuteronEnergy >= 10000)
            deuteronEnergy[lowIndices] = 20
            deuteronEnergy[highIndices] = 10000
        return self.ddnXSfunc(deuteronEnergy)