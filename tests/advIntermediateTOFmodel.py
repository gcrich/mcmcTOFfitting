#!/bin/python
#
# advIntermediateTOFmodel.py
# created winter 2016 g.c.rich
#
# this is a mildly complicated TOF fitting routine which makes fake data
#
# incorporates DDN cross-section weighting, beam-profile spreading, AND
#   Bethe-formula-based deuteron energy loss
#
#
# TODO: add effective posterior predictive check (PPC)
#   that is, sample from the posteriors and produce some fake data
#   compare this fake data with observed data
# TODO: protect against <0 deuteron energy samples?
#

from __future__ import print_function
import numpy as np
from numpy import inf
import scipy.optimize as optimize
from scipy.integrate import odeint
import matplotlib.pyplot as plot
import emcee
import csv as csvlib
import argparse
from constants.constants import (masses, distances, physics, tofWindows)
from utilities.utilities import (beamTimingShape, ddnXSinterpolator, 
                                 getDDneutronEnergy)
from utilities.utilities import readMultiStandoffTOFdata
from utilities.ionStopping import ionStopping


argParser = argparse.ArgumentParser()
argParser.add_argument('-run',choices=[0,1,2,3],default=0,type=int)   
parsedArgs = argParser.parse_args()
runNumber = parsedArgs.run
standoff = {0: distances.tunlSSA_CsI.standoffMid, 
            1: distances.tunlSSA_CsI.standoffClose,
            2: distances.tunlSSA_CsI.standoffClose,
            3: distances.tunlSSA_CsI.standoffFar}
standoffName = {0: 'mid', 1:'close', 2:'close', 3:'far'}

tofWindowSettings = tofWindows()
##############
# vars for binning of TOF 
# this range covers each of the 4 multi-standoff runs
#tof_nBins = 120
#tof_minRange = 130.0
#tof_maxRange = 250.0
tof_nBins = tofWindowSettings.nBins[standoffName[runNumber]]
tof_minRange = tofWindowSettings.minRange[standoffName[runNumber]]
tof_maxRange = tofWindowSettings.maxRange[standoffName[runNumber]]
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

# parameters for making the fake data...
nEvPerLoop = 100000
data_x = np.repeat(x_binCenters,nEvPerLoop)


# PARAMETER BOUNDARIES
min_e0, max_e0 = 600.0,1100.0
min_sigma_0,max_sigma_0 = 0.02, 0.3



ddnXSinstance = ddnXSinterpolator()
beamTiming = beamTimingShape()

# stopping power model and parameters
stoppingMedia_Z = 1
stoppingMedia_A = 2
stoppingMedia_rho = 8.565e-5 # from red notebook, p 157
incidentIon_charge = 1
stoppingMedia_meanExcitation = 19.2
stoppingModelParams = [stoppingMedia_Z, stoppingMedia_A, stoppingMedia_rho,
                       incidentIon_charge, stoppingMedia_meanExcitation]
stoppingModel = ionStopping.simpleBethe( stoppingModelParams )

    
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
    
    
def generateModelData(params, standoffDistance, ddnXSfxn, dedxfxn,
                      nSamples, getPDF=False):
    """
    Generate model data with cross-section weighting applied
    ddnXSfxn is an instance of the ddnXSinterpolator class -
    dedxfxn is a function used to calculate dEdx -
    probably more efficient to these in rather than reinitializing
    one each time
    """
    e0, sigma0 = params
    dataHist = np.zeros((x_bins, eD_bins))
    nLoops = int(nSamples / nEvPerLoop)
    for loopNum in range(0, nLoops):
        eZeros = np.random.normal( e0, sigma0*e0, nEvPerLoop )
        data_eD_matrix = odeint( dedxfxn, eZeros, x_binCenters )
        data_eD = data_eD_matrix.flatten('K')
        data_weights = ddnXSfxn.evaluate(data_eD)
#    print('length of data_x {} length of data_eD {} length of weights {}'.format(
#          len(data_x), len(data_eD), len(data_weights)))
        dataHist2d, xedges, yedges = np.histogram2d( data_x, data_eD,
                                                [x_bins, eD_bins],
                                                [[x_minRange,x_maxRange],[eD_minRange,eD_maxRange]],
                                                weights=data_weights)
        dataHist = np.add(dataHist, dataHist2d)
            
#    print('linalg norm value {}'.format(np.linalg.norm(dataHist)))
#    dataHist = dataHist / np.linalg.norm(dataHist)
#    print('sum of data hist {}'.format(np.sum(dataHist*eD_binSize*x_binSize)))
    dataHist = dataHist/ np.sum(dataHist*eD_binSize*x_binSize)
#    plot.matshow(dataHist)
#    plot.show()
    drawHist2d = (np.rint(dataHist * nSamples)).astype(int)
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



def lnlike(params, observables, nDraws=10000000):
    """
    Evaluate the log likelihood using xs-weighting
    """
    e0, sigma0 = params
    evalDataRaw = generateModelData(params, standoff[runNumber],
                                    ddnXSinstance, stoppingModel.dEdx,
                                    nDraws, True)
    evalData = beamTiming.applySpreading( evalDataRaw )
    logEvalHist = np.log(evalData)
    zeroObservedIndices = np.where(observables == 0)[0]
    for idx in zeroObservedIndices:
        if logEvalHist[idx] == -inf:
            logEvalHist[zeroObservedIndices] = 0

    loglike = np.dot(logEvalHist,observables)
    return loglike
    
    

def lnprior(theta):
    e_0, sigma_0 = theta
    if (min_e0 < e_0 < max_e0 and min_sigma_0 < sigma_0 < max_sigma_0):
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
# *_guess are guesses to start with
mp_e0_guess = 850 # initial deuteron energy, in keV
mp_sigma_0_guess = 0.2 # width of initial deuteron energy spread



# get the data from file
tofData = readMultiStandoffTOFdata('/home/gcr/particleyShared/quenchingFactors/tunlCsI_Jan2016/data/CODA/data/multistandoff.dat')


binEdges = tofData[:,0]

#observedTOF, observed_bin_edges = np.histogram(fakeData[:,3],
#                                               tof_nBins, tof_range)
observedTOF = tofData[:,runNumber+1][(binEdges >= tof_minRange) & (binEdges < tof_maxRange)]
observedTOFbinEdges = tofData[:,0][(binEdges>=tof_minRange)&(binEdges<tof_maxRange)]




# generate fake data
nSamples = 100000
fakeData = generateModelData([mp_e0_guess, mp_sigma_0_guess],
                              standoff[runNumber], 
                              ddnXSinstance, stoppingModel.dEdx, nSamples)



# plot the fake data...
# but only 2000 points, no need to do more
#plot.figure()
#plot.scatter(fakeData[:2000,0], fakeData[:2000,2], color='k', alpha=0.3)
#plot.xlabel('Cell location (cm)')
#plot.ylabel('Neutron energy (keV)')
#plot.draw()


# plot the TOF
tofbins = np.linspace(tof_minRange, tof_maxRange, tof_nBins)
plot.figure()
plot.subplot(211)
plot.scatter(tofbins, observedTOF,color='green')
plot.subplot(212)
plot.scatter(tofbins, fakeData, color='red')
plot.ylabel('counts')
plot.xlabel('TOF (ns)')
plot.xlim(tof_minRange,tof_maxRange)
plot.draw()

plot.show()
# plot the TOF vs x location
# again only plot 2000 points
#plot.figure()
#plot.scatter(fakeData[:2000,2],fakeData[:2000,3], color='k', alpha=0.3)
#plot.xlabel('Neutron energy (keV)' )
#plot.ylabel('TOF (ns)')
#plot.draw()

##########################################
# here's where things are going to get interesting...
# in order to do MCMC, we are going to have to have a log probability fxn
# this means, we need a log LIKELIHOOD function, and this means we
# need just a regular old pdf
# unfortunately, even a regular old PDF is a hideously complicated thing
# no real chance of an analytical approach
# but we can NUMERICALLY attempt to do things

nll = lambda *args: -lnlike(*args)


testNLL = nll([mp_e0_guess, mp_sigma_0_guess], fakeData)
print('test NLL has value {}'.format(testNLL))


parameterBounds=[(min_e0,max_e0),(min_sigma_0,max_sigma_0)]
#minimizedNLL = optimize.minimize(nll, [mp_e0_guess,
#                                       mp_e1_guess, mp_e2_guess, 
#                                       mp_e3_guess, mp_sigma_0_guess,
#                                       mp_sigma_1_guess], 
#                                       args=observedTOF, method='TNC',
#                                       tol=1.0,  bounds=parameterBounds)
#
#print(minimizedNLL)


nDim, nWalkers = 2, 20

#e0, e1, e2, e3, sigma0, sigma1 = minimizedNLL["x"]
e0, sigma0 = mp_e0_guess, mp_sigma_0_guess

p0 = [np.array([e0 + 10 * np.random.randn(), sigma0 + 1e-2 * np.random.randn()]) for i in range(nWalkers)]
sampler = emcee.EnsembleSampler(nWalkers, nDim, lnprob, 
                                kwargs={'observables': observedTOF},
                                threads=12)


fout = open('burninchain.dat','w')
fout.close()

burninSteps = 40
for burninPos, burninProb, burninState in sampler.sample(p0, burninSteps):
    f = open("burninchain.dat", "a")
    for k in range(burninPos.shape[0]):
        fout.write("{0:4d} {1:s}\n".format(k, " ".join(burninPos[k])))
    fout.close()

# save an image of the burn in sampling
plot.figure()
plot.subplot(211)
plot.plot(sampler.chain[:,:,0].T,'-',color='k',alpha=0.2)
plot.ylabel(r'$E_0$ (keV)')
plot.subplot(212)
plot.plot(sampler.chain[:,:,1].T,'-',color='k',alpha=0.2)
plot.ylabel(r'$\sigma_0$ (keV)')
plot.xlabel('Step')
plot.savefig('emceeBurninSampleChainsOut.png',dpi=300)


#sampler.run_mcmc(p0, 500)
# run with progress updates..
fout = open('mainchain.dat','w')
fout.close()

sampler.reset()
mcIterations = 50
for i, samplerResult in enumerate(sampler.sample(burninPos, rstate0=burninState, iterations=mcIterations)):
    if (i+1)%2 == 0:
        print("{0:5.1%}".format(float(i)/mcIterations))
    fout = open('mainchain.dat','a')
    pos=samplerResult[0]
    for k in range(pos.shape[0]):
        fout.write("{0:4d} {1:s}\n".format(k, " ".join(pos[k])))
    fout.close()

plot.figure()
plot.subplot(211)
plot.plot(sampler.chain[:,:,0].T,'-',color='k',alpha=0.2)
plot.ylabel(r'$E_0$ (keV)')
plot.subplot(212)
plot.plot(sampler.chain[:,:,1].T,'-',color='k',alpha=0.2)
plot.ylabel(r'$\sigma_0$ (keV)')
plot.xlabel('Step')
plot.savefig('emceeRunSampleChainsOut.png',dpi=300)
plot.draw()


#samples = sampler.chain[:,400:,:].reshape((-1,nDim))
samples = sampler.chain[:,:,:].reshape((-1,nDim))
# Compute the quantiles.
# this comes from https://github.com/dfm/emcee/blob/master/examples/line.py
e0_mcmc, sigma_0_mcmc = map(lambda v: (v[1], v[2]-v[1],
                                                                v[1]-v[0]),
                                                     zip(*np.percentile(samples, [16, 50, 84],
                                                                        axis=0)))
print("""MCMC result:
    E0 = {0[0]} +{0[1]} -{0[2]}
    sigma_0 = {1[0]} +{1[1]} -{1[2]}
    """.format(e0_mcmc, sigma_0_mcmc))


import corner as corn
cornerFig = corn.corner(samples,labels=["$E_0$","$\sigma_0$"],
                        quantiles=[0.16,0.5,0.84], show_titles=True,
                        title_kwargs={'fontsize': 12})
cornerFig.savefig('emceeRunCornerOut.png',dpi=300)

plot.show()
