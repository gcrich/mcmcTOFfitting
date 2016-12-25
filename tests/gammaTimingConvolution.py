#
#
# gammaTimingConvolution.py
# created winter 2016 g.c.rich
#
# based initially on simpleTOFmodel, this tests the convolution of the gamma
# peak timing characteristics with the generated TOF likelihood


import numpy as np
from numpy import inf
from scipy.integrate import quad
import scipy.optimize as optimize
import matplotlib.pyplot as plot
import emcee
from constants.constants import (masses, qValues, distances, physics)
from utilities.utilities import beamTimingShape


##############
# vars for binning of TOF 
tof_nBins = 25
tof_minRange = 175.0
tof_maxRange = 200.0
tof_range = (tof_minRange,tof_maxRange)


def getDDneutronEnergy(deuteronEnergy, labAngle = 0):
    """Get the energy of neutrons produced by DDN reaction
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
    """Compute time of flight, in nanoseconds, given\
    mass of particle (in keV/c^2), the particle's energy (in keV),\
    and the distance traveled (in cm).
    Though simple enough to write inline, this will be used often.
    """
    velocity = physics.speedOfLight * np.sqrt(2 * energy / mass)
    tof = distance / velocity
    return tof
    
    
def generateModelData(params, nSamples):
    """Generate some fake data from our mode with num samples = nSamples
    params is an array of the parameters, [e0, e1, sigma]
    Returns a tuple of len(nSamples), [x, ed, en, tof]
    """
    initialEnergy, eLoss, sigma = params
    data_x=np.random.uniform(low=0.0, high=distances.tunlSSA_CsI.cellLength, 
                             size=nSamples)
    data_ed= np.random.normal(loc=initialEnergy + eLoss*data_x, 
                              scale=sigma)
    data_en = getDDneutronEnergy(data_ed)
    
    neutronDistance = distances.tunlSSA_CsI.cellToZero + (distances.tunlSSA_CsI.cellLength - data_x)
    neutronTOF = getTOF(masses.neutron, data_en, neutronDistance)
    effectiveDenergy = (initialEnergy + data_ed)/2
    deuteronTOF = getTOF( masses.deuteron, effectiveDenergy, data_x )
    data_tof = neutronTOF + deuteronTOF
    
    data = np.column_stack((data_x,data_ed,data_en,data_tof))
    return data
    
def lnlike(params, observables, nDraws=1000000):
    """Evaluate the log likelihood given a set of params and observables
    Observables is a vector; a histogrammed time distribution
    Params is a list of [initial D energy, E_D loss, sigma]
    nDraws is the number of points drawn from (energy,location) distribution\
    which are used to produce the PDF evaluations at different TOFs
    """
    #print('checking type ({}) and length ({}) of params in lnlikefxn'.format(type(params),len(params)))
    evalData=generateModelData(params, nDraws)
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
    e_0, e_1, sigma = theta
    if 800 < e_0 < 1200 and -200 <e_1<0 and 10 < sigma < 100:
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
   
# mp_* are model parameters
# *_t are 'true' values that go into our fake data
mp_initialEnergy_t = 1100 # initial deuteron energy, in keV
mp_loss0_t = -100 # energy loss, 0th order approx, in keV/cm
mp_sigma_t = 50 # width of deuteron energy spread, fixed for now, in keV


# generate fake data
nSamples = 10000
fakeData = generateModelData([mp_initialEnergy_t,mp_loss0_t,mp_sigma_t],
                             nSamples)






nll = lambda *args: -lnlike(*args)


observedTOF, observed_bin_edges = np.histogram(fakeData[:,3], 
                                               tof_nBins, tof_range)
beamTiming = beamTimingShape()
spreadData = beamTiming.applySpreading( observedTOF )

print('length of raw array {} and spread array {}'.
      format(len(observedTOF), len(spreadData)))

plot.figure()
plot.scatter(observed_bin_edges[:-1], observedTOF,color='red')
plot.scatter(observed_bin_edges[:-1], spreadData,color='green')
plot.show()


        
