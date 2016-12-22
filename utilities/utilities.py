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
import scipy.optimize as optimize
import csv as csvlib
import constants.constants.distances.tunlSSA_CsI as distances
import constants.constants.physics
from constants.constants import (masses, qValues)



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
    velocity = speedOfLight * np.sqrt(2 * energy / mass)
    tof = distance / velocity
    return tof
    
    
def generateModelData(params, standoffDistance, nSamples):
    """
    Generate some fake data from our mode with num samples = nSamples
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
    """
    Read in data from TAC for multiple-standoff runs
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
