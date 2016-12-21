#
#
# xsWeightedSampling.py
# created winter 2016 g.c.rich
#
# tests for cross-section weighted sampling of the DDn production distribution
#

import numpy as np
from numpy import inf
from scipy.integrate import quad
import scipy.optimize as optimize
import matplotlib.pyplot as plot
import emcee
from scipy.interpolate import interp1d

#
# CONSTANTS
#
# these are perhaps presently not included in a very pythonic way
# but to get going, here we are..
#
speedOfLight = 29.9792 # in cm/ns
mass_deuteron = 1.8756e+06 # keV /c^2
mass_neutron = 939565.0 # keV/c^2
mass_he3 = 2.809414e6 # keV/c^2

# Q value of DDN reaction, in keV
qValue_ddn = 3268.914


distance_cellToZero = 518.055 # cm, distance from tip of gas cell to 0deg face
distance_cellLength = 2.86 # cm, length of gas cell
distance_zeroDegLength = 3.81 # cm, length of 0deg detector
distance_tipToColli = 148.4 # cm, distance from cell tip to collimator exit
distance_colliToZero = 233.8 # cm, distance from collimator exit to face of
# 0deg AT ITS CLOSEST LOCATION
distance_delta1 = 131.09 # cm, difference between close and mid 0degree loc.
distance_delta2 = 52.39 # cm, difference between mid and far 0deg loc

# in function generateModelData the standoff distance has 0deg length / 2 added
distance_standoffClose= distance_tipToColli + distance_colliToZero
distance_standoffMid = distance_standoffClose + distance_delta1
distance_standoffFar = distance_standoffMid + distance_delta2

##############
# vars for binning of TOF 
tof_nBins = 25
tof_minRange = 180.0
tof_maxRange = 205.0
tof_range = (tof_minRange,tof_maxRange)


# PARAMETER BOUNDARIES
min_e0, max_e0 = 900,1100
min_e1,max_e1= -100, 0
min_e2,max_e2 = -30, 0
min_e3,max_e3 = -10, 0
min_sigma,max_sigma = 40, 100


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
        return self.ddnXSfunc(deuteronEnergy)
    


def getDDneutronEnergy(deuteronEnergy, labAngle = 0):
    """Get the energy of neutrons produced by DDN reaction
    Function accepts the deuteron energy (in keV) and the angle (in lab\
    frame of reference) at which the neutron is emitted.
    Returns neutron energy in keV
    """     
    neutronAngle_radians = labAngle * np.pi / 180
    rVal = np.sqrt(mass_deuteron * mass_neutron*deuteronEnergy) / \
                   (mass_neutron + mass_he3) * \
                   np.cos(neutronAngle_radians)
    sVal = (deuteronEnergy *( mass_he3 - mass_deuteron) +
            qValue_ddn * mass_he3) / (mass_neutron + mass_he3)
    sqrtNeutronEnergy = rVal + np.sqrt(np.power(rVal,2) + sVal)
    return np.power(sqrtNeutronEnergy, 2)
    
def getTOF(mass, energy, distance):
    """Compute time of flight, in nanoseconds, given\
    mass of particle (in keV/c^2), the particle's energy (in keV),\
    and the distance traveled (in cm).
    Though simple enough to write inline, this will be used often.
    """
    velocity = speedOfLight * np.sqrt(2 * energy / mass)
    tof = distance / velocity
    return tof
    
    
def generateModelData(params, standoffDistance, nSamples):
    """Generate some fake data from our mode with num samples = nSamples
    params is an array of the parameters, [e0, e1, sigma]
    Returns a tuple of len(nSamples), [x, ed, en, tof]
    """
    initialEnergy, eLoss, e2, e3, sigma = params
    
    data_x=np.random.uniform(low=0.0, high=distance_cellLength, size=nSamples)
    meanEnergy = initialEnergy + eLoss*data_x + \
                              e2*np.power(data_x,2) + e3 * np.power(data_x,3)
    data_ed= np.random.normal(loc=meanEnergy, scale=sigma)
    data_en = getDDneutronEnergy(data_ed)
    
    neutronDistance = standoffDistance + (distance_cellLength - data_x) + \
                        distance_zeroDegLength/2
    neutronTOF = getTOF(mass_neutron, data_en, neutronDistance)
    effectiveDenergy = (initialEnergy + data_ed)/2
    deuteronTOF = getTOF( mass_deuteron, effectiveDenergy, data_x )
    data_tof = neutronTOF + deuteronTOF
    
    data = np.column_stack((data_x,data_ed,data_en,data_tof))
    return data
    
def lnlike(params, observables, nDraws=1000000):
    """Evaluate the log likelihood given a set of params and observables
    Observables is a vector a histogrammed time distribution
    Params is a list of [initial D energy, E_D loss, sigma]
    nDraws is the number of points drawn from (energy,location) distribution\
    which are used to produce the PDF evaluations at different TOFs
    """
    #print('checking type ({}) and length ({}) of params in lnlikefxn'.format(type(params),len(params)))
    evalData=generateModelData(params, distance_standoffMid, nDraws)
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
    e_0, e_1, e_2, e_3, sigma = theta
    if 900 < e_0 < 1100 and -100 <e_1<0 and -30 < e_2 < 0 \
    and -10 < e_3 < 0 and 40 < sigma < 100:
        return 0
    return -inf
    
def lnprob(theta, observables):
    """Evaluate the log probability
    theta is a list of the model parameters
    observables is, in this case, a histogram of TOF values
    """
    prior = lnprior(theta)
    if not np.isfinite(prior):
        return -inf
    return prior + lnlike(theta, observables)
   
    
def getDenergyAtLocation(params, cellLocation):
    """Get the deuteron energy at a cell location, given model params"""
    e0, e1, e2, e3, sigma = params
    meanEnergy = e0 + e1*cellLocation + \
                              e2*np.power(cellLocation,2) + \
                              e3 * np.power(cellLocation,3)
    return meanEnergy
    
    
# mp_* are model parameters
# *_t are 'true' values that go into our fake data
# *_guess are guesses to start with
mp_e0_guess = 1050 # initial deuteron energy, in keV
mp_e1_guess = -80 # energy loss, 0th order approx, in keV/cm
mp_e2_guess = -10
mp_e3_guess = -5
mp_sigma_guess = 80 # width of deuteron energy spread, fixed for now, in keV




# generate fake data
nSamples = 100000
fakeData = generateModelData([mp_e0_guess,mp_e1_guess,mp_e2_guess, 
                              mp_e3_guess, mp_sigma_guess], 
                              distance_standoffMid, nSamples)



# plot the fake data...
# but only 2000 points, no need to do more
plot.figure()
plot.scatter(fakeData[:2000,0], fakeData[:2000,1], color='k', alpha=0.3)
plot.xlabel('Cell location (cm)')
plot.ylabel('Deuteron energy (keV)')
plot.show()


# plot the TOF 
plot.figure()
plot.hist(fakeData[:,3], 25, (180,205))
plot.ylabel('counts')
plot.show()

# plot the TOF vs x location
# again only plot 2000 points
plot.figure()
plot.scatter(fakeData[:2000,2],fakeData[:2000,3], color='k', alpha=0.3)
plot.xlabel('Neutron energy (keV)' )
plot.ylabel('TOF (ns)')
plot.show()

# make a plot that compares the spline to the data points from which
# we form it
ddnXS= ddnXSinterpolator()
xsSplineRatio = ddnXS.evaluate(ddnXS.dEnergies) / ddnXS.ddnSigmaZero
xpoints = np.linspace(20,10000,num=200)
plot.figure()
plot.subplot(211)
plot.scatter(ddnXS.dEnergies, ddnXS.ddnSigmaZero)
plot.plot(xpoints, ddnXS.evaluate(xpoints))
plot.xlim(0,1200)
plot.ylim(0, 24)
plot.ylabel('D(D,n) cross section (mb)')
plot.subplot(212)
plot.scatter(ddnXS.dEnergies, xsSplineRatio)
plot.xlim(0, 1200)
plot.ylim(0.99,1.01)
plot.xlabel('Deuteron energy (keV)')
plot.show()

nBins = 100
uniformLengthSamples = np.random.uniform(0., distance_cellLength, nSamples)
binSize = distance_cellLength / nBins
binCenters, binSize = np.linspace(binSize/2, distance_cellLength - binSize/2, 
                                  100, retstep =True)
energySamples = getDenergyAtLocation([mp_e0_guess, mp_e1_guess, mp_e2_guess, 
                                 mp_e3_guess, mp_sigma_guess], 
                                 uniformLengthSamples)
xsWeights = ddnXS.evaluate(energySamples)
lengthSamplesHist, binEdges = np.histogram(uniformLengthSamples, 100,
                                           (0., distance_cellLength), 
                                           weights = xsWeights, 
                                           density =True )
integratedXSweightedPDF = np.sum(lengthSamplesHist*binSize)
print('integral of the XS weighted PDF along length is {}'.format(
      integratedXSweightedPDF))
plot.figure()
plot.scatter(binCenters, lengthSamplesHist*binSize)
plot.ylim(0.006,0.014)
plot.show()