#
#
# intermediateTOFfit.py
# created winter 2016 g.c.rich
#
# this is a mildly complicated TOF fitting routine
#
# incorporates DDN cross-section weighting and beam-profile spreading
#
# TODO: add effective posterior predictive check (PPC)
#   that is, sample from the posteriors and produce some fake data
#   compare this fake data with observed data
from __future__ import print_function
import numpy as np
from numpy import inf
from scipy.integrate import quad
import scipy.optimize as optimize
import matplotlib.pyplot as plot
import emcee
import csv as csvlib
from constants.constants import (masses, qValues, distances, physics)
from utilities.utilities import beamTimingShape


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



beamTiming = beamTimingShape()


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
    
    
def generateModelData(params, standoffDistance, ddnXSfxn, nSamples, getPDF=False):
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
    """
    Evaluate the log likelihood using xs-weighting
    """
    e0, e1,e2,e3,sigma = params
    evalDataRaw = generateModelData_XS(params, distances.tunlSSA_CsI.standoffMid,
                                          ddnXSinstance, nDraws, True)
    evalData = beamTiming.applySpreading( evalDataRaw )
    logEvalHist = np.log(evalData)
    zeroObservedIndices = np.where(observables == 0)[0]
    for idx in zeroObservedIndices:
        if logEvalHist[idx] == -inf:
            logEvalHist[zeroObservedIndices] = 0

    loglike = np.dot(logEvalHist,observables)
    return loglike
    
    

def lnprior(theta):
    e_0, e_1, e_2, e_3, sigma = theta
    if min_e0 < e_0 < max_e0 and min_e1 <e_1< max_e1 and min_e2 < e_2<max_e2 \
    and min_e3 < e_3 < max_e3 and min_sigma < sigma < max_sigma:
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
    
    
# mp_* are model parameters
# *_t are 'true' values that go into our fake data
# *_guess are guesses to start with
mp_e0_guess = 950 # initial deuteron energy, in keV
mp_e1_guess = -100 # energy loss, 0th order approx, in keV/cm
mp_e2_guess = -10
mp_e3_guess = -5
mp_sigma_guess = 80 # width of deuteron energy spread, fixed for now, in keV


# get the data from file
tofData = readMultiStandoffTOFdata('/home/gcr/particleyShared/quenchingFactors/tunlCsI_Jan2016/data/CODA/data/multistandoff.dat')


binEdges = tofData[:,0]

#observedTOF, observed_bin_edges = np.histogram(fakeData[:,3], 
#                                               tof_nBins, tof_range)
observedTOF = tofData[:,1][(binEdges >= 180) & (binEdges < 205)]
observedTOFbinEdges = tofData[:,0][(binEdges>=180)&(binEdges<205)]



plot.figure()
plot.scatter(binEdges, tofData[:,1] )
plot.xlabel('tof (ns)')
plot.ylabel('counts')
plot.xlim(170.0,220.0)
plot.draw()

# generate fake data
nSamples = 10000
fakeData = generateModelData([mp_e0_guess,mp_e1_guess,mp_e2_guess, 
                              mp_e3_guess, mp_sigma_guess], 
                              distance_standoffMid, nSamples)



# plot the fake data...
# but only 2000 points, no need to do more
plot.figure()
plot.scatter(fakeData[:2000,0], fakeData[:2000,2], color='k', alpha=0.3)
plot.xlabel('Cell location (cm)')
plot.ylabel('Neutron energy (keV)')
plot.draw()


# plot the TOF 
plot.figure()
plot.subplot(211)
plot.hist(fakeData[:,3], 25, (180,205))
plot.ylabel('counts')
plot.subplot(212)
plot.scatter(observedTOFbinEdges,observedTOF)
plot.xlim(180,205)
plot.xlabel('TOF (ns)')
plot.ylabel('counts')
plot.draw()

# plot the TOF vs x location
# again only plot 2000 points
plot.figure()
plot.scatter(fakeData[:2000,2],fakeData[:2000,3], color='k', alpha=0.3)
plot.xlabel('Neutron energy (keV)' )
plot.ylabel('TOF (ns)')
plot.draw()

##########################################
# here's where things are going to get interesting...
# in order to do MCMC, we are going to have to have a log probability fxn
# this means, we need a log LIKELIHOOD function, and this means we
# need just a regular old pdf
# unfortunately, even a regular old PDF is a hideously complicated thing
# no real chance of an analytical approach
# but we can NUMERICALLY attempt to do things

nll = lambda *args: -lnlike(*args)


testNLL = nll([mp_e0_guess, mp_e1_guess, mp_e2_guess, 
               mp_e3_guess, mp_sigma_guess], observedTOF)
print('test NLL has value {}'.format(testNLL))


parameterBounds=[(min_e0,max_e0),(min_e1,max_e1),(min_e2,max_e2),
                 (min_e3,max_e3),(min_sigma,max_sigma)]
minimizedNLL = optimize.minimize(nll, [mp_e0_guess,
                                       mp_e1_guess, mp_e2_guess, 
                                       mp_e3_guess, mp_sigma_guess], 
                                       args=observedTOF, method='TNC',
                                       tol=1.0,  bounds=parameterBounds)

print(minimizedNLL)


nDim, nWalkers = 5, 100

e0, e1, e2, e3, sigma = minimizedNLL["x"]

p0 = [[e0,e1,e2,e3,sigma] + 1e-1 * np.random.randn(nDim) for i in range(nWalkers)]
sampler = emcee.EnsembleSampler(nWalkers, nDim, lnprob, 
                                kwargs={'observables': observedTOF}, 
                                threads=8)

#sampler.run_mcmc(p0, 500)
# run with progress updates..
mcIterations = 5000
for i, samplerResult in enumerate(sampler.sample(p0, iterations=mcIterations)):
    if (i+1)%50 == 0:
        print("{0:5.1%}".format(float(i)/mcIterations))

plot.figure()
plot.subplot(511)
plot.plot(sampler.chain[:,:,0].T,'-',color='k',alpha=0.2)
plot.ylabel('Initial energy (keV)')
plot.subplot(512)
plot.plot(sampler.chain[:,:,1].T,'-',color='k',alpha=0.2)
plot.ylabel('Energy loss (keV/cm)')
plot.subplot(513)
plot.plot(sampler.chain[:,:,2].T,'-',color='k',alpha=0.2)
plot.ylabel(r'$E_2$ (keV/cm$^2$)')
plot.xlabel('Step')
plot.subplot(514)
plot.plot(sampler.chain[:,:,3].T,'-',color='k',alpha=0.2)
plot.ylabel(r'$E_3$ (keV/cm$^3$)')
plot.xlabel('Step')
plot.subplot(515)
plot.plot(sampler.chain[:,:,4].T,'-',color='k',alpha=0.2)
plot.ylabel('Sigma (keV)')
plot.xlabel('Step')
plot.draw()

# Compute the quantiles.
# this comes from https://github.com/dfm/emcee/blob/master/examples/line.py
e0_mcmc, e1_mcmc, e2_mcmc, e3_mcmc, sigma_mcmc = map(lambda v: (v[1], v[2]-v[1],
                                                                v[1]-v[0]),
                                                     zip(*np.percentile(samples, [16, 50, 84],
                                                                        axis=0)))
print("""MCMC result:
    E0 = {0[0]} +{0[1]} -{0[2]}
    E1 = {1[0]} +{1[1]} -{1[2]}
    E2 = {2[0]} +{2[1]} -{2[2]}
    E3 = {3[0]} +{3[1]} -{3[2]}
    sigma = {4[0]} +{4[1]} -{4[2]}
    """.format(e0_mcmc, mp_e0_t, e1_mcmc, mp_e1_t,
               sigma_mcmc, mp_sigma_t))

samples = sampler.chain[:,2000:,:].reshape((-1,nDim))
import corner as corn
cornerFig = corn.corner(samples,labels=["$E_0$","$E_1$","$E_2$","$E_3$",
                                        "$\sigma$"],
                        quantiles=[0.16,0.5,0.84], show_titles=True,
                        title_kwargs={'fontsize': 12})


plot.show()