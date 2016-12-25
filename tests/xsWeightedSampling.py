#
#
# xsWeightedSampling.py
# created winter 2016 g.c.rich
#
# tests for cross-section weighted sampling of the DDn production distribution
#
from __future__ import print_function
import numpy as np
from numpy import inf
from scipy.integrate import quad
import scipy.optimize as optimize
import matplotlib.pyplot as plot
import emcee
from scipy.interpolate import interp1d
from constants.constants import (masses, qValues, distances, physics)
from utilities.utilities import ddnXSinterpolator


##############
# vars for binning of TOF 
tof_nBins = 60
tof_minRange = 160.0
tof_maxRange = 220.0
tof_range = (tof_minRange,tof_maxRange)

eD_bins = 150
eD_minRange = 200.0
eD_maxRange = 1700.0
eD_range = (eD_minRange, eD_maxRange)
eD_binSize = (eD_maxRange - eD_minRange)/eD_bins
eD_binCenters = np.linspace(eD_minRange + eD_binSize/2,
                            eD_maxRange - eD_binSize/2,
                            eD_bins)


x_bins = 100
x_minRange = 0.0
x_maxRange = distances.tunlSSA_CsI.cellLength
x_range = (x_minRange,x_maxRange)
x_binSize = (x_maxRange - x_minRange)/x_bins
x_binCenters = np.linspace(x_minRange + x_binSize/2,
                           x_maxRange - x_binSize/2,
                           x_bins)


# PARAMETER BOUNDARIES
min_e0, max_e0 = 900,1100
min_e1,max_e1= -100, 0
min_e2,max_e2 = -30, 0
min_e3,max_e3 = -10, 0
min_sigma,max_sigma = 40, 100


ddnXSinstance = ddnXSinterpolator()
    


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


eN_binCenters = getDDneutronEnergy( eD_binCenters )

    
def getTOF(mass, energy, distance):
    """Compute time of flight, in nanoseconds, given\
    mass of particle (in keV/c^2), the particle's energy (in keV),\
    and the distance traveled (in cm).
    Though simple enough to write inline, this will be used often.
    """
    velocity = physics.speedOfLight * np.sqrt(2 * energy / mass)
    tof = distance / velocity
    return tof
    
    
def generateModelData(params, standoffDistance, nSamples):
    """Generate some fake data from our mode with num samples = nSamples
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

def generateModelData_XS(params, standoffDistance, ddnXSfxn, nSamples, getPDF=False):
    """
    Generate model data with cross-section weighting applied
    ddnXSfxn is an instance of the ddnXSinterpolator class - 
    probably more efficient to pass it in rather than reinitializing 
    one each time
    """
    e0, e1, e2, e3, sigma = params
    data_x=np.random.uniform(low=0.0, high=distances.tunlSSA_CsI.cellLength, size=nSamples)

    meanEnergy = (e0 + e1*data_x + e2*np.power(data_x,2) +
                  e3 * np.power(data_x,3))
    data_eD = np.random.normal(loc=meanEnergy, scale=sigma)
    data_weights = ddnXSfxn.evaluate(data_eD)
    dataHist2d, xedges, yedges = np.histogram2d( data_x, data_eD,
                                                [x_bins, eD_bins],
                                                [[x_minRange,x_maxRange],[eD_minRange,eD_maxRange]],
                                                normed=True,
                                                weights=data_weights)
    drawHist2d = (np.rint(dataHist2d * eD_binSize * x_binSize * nSamples)).astype(int)
    tofs = []
    tofWeights = []
    for index, weight in np.ndenumerate( drawHist2d ):
        cellLocation = x_binCenters[index[0]]
        effectiveDenergy = (e0 + eD_binCenters[index[1]])/2
        tof_d = getTOF( masses.deuteron, effectiveDenergy, cellLocation )
        neutronDistance = (distances.tunlSSA_CsI.cellLength - cellLocation +
                           distances.tunlSSA_CsI.zeroDegLength/2 +
                           standoffDistance )
        tof_n = getTOF(masses.neutron, eN_binCenters[index[1]], neutronDistance)
        tofs.append( tof_d + tof_n )
        tofWeights.append(weight)
    tofData, tofBinEdges = np.histogram( tofs, bins=tof_nBins, range=tof_range,
                           weights=tofWeights, density=getPDF)
    return tofData

def lnlike(params, observables, nDraws=1000000):
    """Evaluate the log likelihood given a set of params and observables
    Observables is a vector a histogrammed time distribution
    Params is a list of [initial D energy, E_D loss, sigma]
    nDraws is the number of points drawn from (energy,location) distribution\
    which are used to produce the PDF evaluations at different TOFs
    """
    #print('checking type ({}) and length ({}) of params in lnlikefxn'.format(type(params),len(params)))
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


def lnlike_xs(params, observables, nDraws=1000000):
    """
    Evaluate the log likelihood using xs-weighting
    """
    e0, e1,e2,e3,sigma = params
    print('likelihood called with params {},{},{},{},{}'.
          format(e0,e1,e2,e3,sigma))
    evalData = generateModelData_XS(params, distances.tunlSSA_CsI.standoffMid,
                                    ddnXSinstance, nDraws, True)
    logEvalHist = np.log(evalData)
    zeroObservedIndices = np.where(observables == 0)[0]
    for idx in zeroObservedIndices:
        if logEvalHist[idx] == -inf:
            logEvalHist[zeroObservedIndices] = 0
    
    loglike = np.dot(logEvalHist,observables)
    print('got liklihood {}'.format(loglike))
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
                              distances.tunlSSA_CsI.standoffMid, nSamples)



# plot the fake data...
# but only 2000 points, no need to do more
#plot.figure(1)
#plot.scatter(fakeData[:2000,0], fakeData[:2000,1], color='k', alpha=0.3)
#plot.xlabel('Cell location (cm)')
#plot.ylabel('Deuteron energy (keV)')
#plot.show()


# plot the TOF 
#plot.figure(2)
#plot.hist(fakeData[:,3], 25, (180,205))
#plot.ylabel('counts')
#plot.show()

# plot the TOF vs x location
# again only plot 2000 points
#plot.figure(3)
#plot.scatter(fakeData[:2000,2],fakeData[:2000,3], color='k', alpha=0.3)
#plot.xlabel('Neutron energy (keV)' )
#plot.ylabel('TOF (ns)')
#plot.show()

# make a plot that compares the spline to the data points from which
# we form it
ddnXS= ddnXSinterpolator()
xsSplineRatio = ddnXS.evaluate(ddnXS.dEnergies) / ddnXS.ddnSigmaZero
xpoints = np.linspace(20,10000,num=200)
#plot.figure(4)
#plot.subplot(211)
#plot.scatter(ddnXS.dEnergies, ddnXS.ddnSigmaZero)
#plot.plot(xpoints, ddnXS.evaluate(xpoints))
#plot.xlim(0,1200)
#plot.ylim(0, 24)
#plot.ylabel('D(D,n) cross section (mb)')
#plot.subplot(212)
#plot.scatter(ddnXS.dEnergies, xsSplineRatio)
#plot.xlim(0, 1200)
#plot.ylim(0.99,1.01)
#plot.xlabel('Deuteron energy (keV)')
#plot.show()

nBins = 100
uniformLengthSamples = np.random.uniform(0., distances.tunlSSA_CsI.cellLength, nSamples)
binSize = distances.tunlSSA_CsI.cellLength / nBins
binCenters, binSize = np.linspace(binSize/2, distances.tunlSSA_CsI.cellLength - binSize/2, 
                                  100, retstep =True)
energySamples = getDenergyAtLocation([mp_e0_guess, mp_e1_guess, mp_e2_guess, 
                                 mp_e3_guess, mp_sigma_guess], 
                                 uniformLengthSamples)
xsWeights = ddnXS.evaluate(energySamples)
lengthSamplesHist, binEdges = np.histogram(uniformLengthSamples, 100,
                                           (0., distances.tunlSSA_CsI.cellLength), 
                                           weights = xsWeights, 
                                           density =True )
integratedXSweightedPDF = np.sum(lengthSamplesHist*binSize)
print('integral of the XS weighted PDF along length is {}'.format(
      integratedXSweightedPDF))
#plot.figure(5)
#plot.scatter(binCenters, lengthSamplesHist*binSize)
#plot.ylim(0.006,0.014)
#plot.show()

# here, we'll test doing uniform sampling then actually generating the gaussian
# spread around the mean
# from THESE points, we'll do the weighting...
meanEnergy = (mp_e0_guess + mp_e1_guess*uniformLengthSamples +
              mp_e2_guess*np.power(uniformLengthSamples,2) +
              mp_e3_guess * np.power(uniformLengthSamples,3))
data_eD = np.random.normal(loc=meanEnergy, scale=mp_sigma_guess)
data_weights = ddnXS.evaluate(data_eD)
hist2d, xedges, yedges = np.histogram2d( uniformLengthSamples, data_eD, [100,eD_bins],
                         [[0.0,distances.tunlSSA_CsI.cellLength],
                          [eD_minRange,eD_maxRange]], normed=True,
                         weights=data_weights)
plot.figure()
plot.matshow(hist2d, origin='lower', interpolation='none')
plot.xlabel('location in cell')
plot.ylabel('deuteron energy')
plot.show()

hist2d_binsize = (distances.tunlSSA_CsI.cellLength *
                  (eD_maxRange-eD_minRange) / (eD_bins*100))
hist2d_integral = np.sum(hist2d*hist2d_binsize)
print('integral of 2d histogram {}'.format(hist2d_integral))
print(hist2d.shape)
projected = hist2d.sum(axis=1)
projectedE = hist2d.sum(axis=0)
print(projected.shape)
#plot.figure()
#plot.subplot(121)
#plot.scatter(xedges[:-1], projected)
#plot.subplot(122)
#plot.scatter(yedges[:-1], projectedE)
#plot.show()
projectedBinsize = distances.tunlSSA_CsI.cellLength / 100
summedProjection = np.sum(projected*hist2d_binsize)
print('integrated projection along cell length {} (should be 1)'.
      format(summedProjection))

# so at this stage, the 2d hist is our PDF..
# now we need to sample from it
hist2d_draws = (np.rint(hist2d * hist2d_binsize * nSamples)).astype(int)
print(type(hist2d_draws))
print(hist2d_draws.shape)
print(type(hist2d_draws[0,0]))
effectiveDraws = hist2d_draws.sum()
print('actual \'draws\' from the histogram {}'.format(effectiveDraws))

# calculate the bin centers
binSize_cellLoc = distances.tunlSSA_CsI.cellLength / nBins
binCenters_cellLoc = np.linspace( binSize_cellLoc/2,
                                 (distances.tunlSSA_CsI.cellLength -
                                  binSize_cellLoc/2), nBins )
binSize_eD = (eD_maxRange - eD_minRange) / eD_bins
binCenters_eD = np.linspace( eD_minRange + binSize_eD/2,
                            eD_maxRange - binSize_eD/2,
                            eD_bins)
binCenters_eN = getDDneutronEnergy( binCenters_eD ) # NOTE THAT THIS IS NOT NECESSARILY A LINEAR TRANSFORM
# SO NOW OUR BINS MAY NOT BE OF EQUAL VOLUME.. or something.. need to think about it
tofs = []
tofWeights = []
for index, weight in np.ndenumerate(hist2d_draws):
# get the TOF for the deuteron to this location
    cellLocation = binCenters_cellLoc[index[0]]
    effectiveDenergy = (mp_e0_guess + binCenters_eD[index[1]]) / 2
    tof_d = getTOF( masses.deuteron, effectiveDenergy, cellLocation)
    neutronDistance = (distances.tunlSSA_CsI.standoffMid +
                       distances.tunlSSA_CsI.cellLength - cellLocation +
                       distances.tunlSSA_CsI.zeroDegLength/2)
    tof_n = getTOF( masses.neutron, binCenters_eN[index[1]], neutronDistance)
    tofs.append( tof_d + tof_n)
    tofWeights.append( weight )
#plot.figure()
#plot.hist(tofs,weights=tofWeights, bins=tof_nBins, range=tof_range)
#plot.show()


fakeXSdata = generateModelData_XS( [mp_e0_guess, mp_e1_guess, mp_e2_guess,
                                    mp_e3_guess, mp_sigma_guess],
                                  distances.tunlSSA_CsI.standoffMid,
                                  ddnXSinstance, 200000 )
loglikeValue = lnlike_xs( [1080, mp_e1_guess*0.92, mp_e2_guess*1.12,
                           mp_e3_guess*0.95, mp_sigma_guess*1.1], fakeXSdata )
print('got likelihood value of: {}'.format(loglikeValue))

nll = lambda *args: -lnlike_xs(*args)
minimizedNLL = optimize.minimize(nll, [1080, mp_e1_guess*0.92, mp_e2_guess*1.12,
                                       mp_e3_guess*0.8, mp_sigma_guess*1.2],
                                       args=fakeXSdata, method='Nelder-Mead',
                                       tol=1.0)

print(minimizedNLL)