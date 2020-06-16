#!/bin/python
#
# csi_oneBD.py
#
# hacked together quick fitting of "one BD" CsI[Na] QF run
# in a time crunch, so .. copied other "test"
# will try to comment as i go, but.. 
# 
# copied from
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
from scipy.integrate import ode
from scipy.interpolate import RectBivariateSpline
from scipy.stats import (poisson, norm, lognorm)
import scipy.special as special
#from scipy.stats import skewnorm
import sys
import emcee
import csv as csvlib
import argparse
from numbers import Number
from constants.constants import (masses, distances, physics, tofWindows)
from utilities.utilities import (beamTimingShape, ddnXSinterpolator, 
                                 getDDneutronEnergy)
from utilities.utilities import zeroDegreeTimingSpread
from utilities.utilities import readMultiStandoffTOFdata
from utilities.ionStopping import ionStopping
from math import isnan
import gc
#from fast_histogram import histogram1d


TESTONLY = True


defaultInputDataFilename = '/Users/grayson/Documents/quenchingFactors/csi_oneBD/tof/oneBD_mcmcInputData.dat'
runColors=['#e41a1c','#377eb8','#4daf4a']#,'#984ea3', '#ff7f00']

argParser = argparse.ArgumentParser()
argParser.add_argument('-run',choices=[0,1,2,3],default=0,type=int)   
argParser.add_argument('-inputDataFilename', default=defaultInputDataFilename, type=str)
argParser.add_argument('-mpi', default=0,type=int)
argParser.add_argument('-debug', choices=[0,1], default=0,type=int)
argParser.add_argument('-nThreads', default=5, type=int)
argParser.add_argument('-quitEarly', choices=[0,1], default=0, type=int)
argParser.add_argument('-batch',choices=[0,1], default=0, type=int)
argParser.add_argument('-forceCustomPDF', choices=[0,1], default=0, type=int)
argParser.add_argument('-nDrawsPerEval', default=200000, type=int) # number of draws from distribution used in each evaluation of the likelihood function
argParser.add_argument('-nBurninSteps', default=400, type=int)
argParser.add_argument('-nMainSteps', default=100, type=int)
argParser.add_argument('-outputPrefix', type=str, default='')
argParser.add_argument('-nWalkers', type=int, default=256)
argParser.add_argument('-qnd', type=int, default=0, choices=[0,1], help='Quick and dirty (0,1): reduce binning behind the scenes')

parsedArgs = argParser.parse_args()
runNumber = parsedArgs.run
inputDataFilename = parsedArgs.inputDataFilename
nMPInodes = parsedArgs.mpi
debugFlag = parsedArgs.debug
nThreads = parsedArgs.nThreads
nDrawsPerEval = parsedArgs.nDrawsPerEval
burninSteps = parsedArgs.nBurninSteps
mcIterations = parsedArgs.nMainSteps
outputPrefix = parsedArgs.outputPrefix
quickAndDirty = True if parsedArgs.qnd == 1 else 0

# batchMode turns off plotting and extraneous stuff like test NLL eval at beginning 
batchMode = False
if parsedArgs.batch == 1:
    batchMode=True

quitEarly = False
if parsedArgs.quitEarly ==1:
    quitEarly= True

forceCustomPDF = False
if parsedArgs.forceCustomPDF == 1:
    forceCustomPDF = True
    
debugging = False
if debugFlag == 1:
    debugging = True

useMPI= False
if nMPInodes > 0:
    useMPI = True
    
if useMPI:
    from emcee.utils import MPIPool

#
# SHOULD PROBABLY PRINT OUT HOW WE ARE RUNNING JUST FOR CLARITY
# for now.. just.. some specific things since god im stressed
#
if quickAndDirty == True:
    print('\n\nRUNNING IN QUICK AND DIRTY MODE\n\n')



####################
# CHECK TO SEE IF WE ARE USING PYTHON VERSION 2.7 OR ABOVE
# if using earlier version, we will NOT have matplotlib
# so, if this is the case, don't do any plotting
doPlotting = True
if batchMode:
    doPlotting = False
if sys.version_info[0] < 3 and sys.version_info[1] < 7:
    print('detected python version {0[0]}.{0[1]}, disabling plotting'.format(sys.version_info))
    doPlotting = False
if doPlotting:
    import matplotlib.pyplot as plot

    
# check for skewnorm distribution in scipy
if forceCustomPDF:
    import utilities.pdfs as utePdfs
    skewnorm = utePdfs.skewnorm()
else:
    try:
        from scipy.stats import skewnorm
    except ImportError:
        print('could not load scipy skewnorm distribution - using our own')
        import utilities.pdfs as utePdfs
        skewnorm = utePdfs.skewnorm()


nRuns = 3

standoff = {0: distances.tunlSSA_CsI_oneBD.standoffClose, 
            1: distances.tunlSSA_CsI_oneBD.standoffMid,
            2: distances.tunlSSA_CsI_oneBD.standoffFar}
standoffs = [distances.tunlSSA_CsI_oneBD.standoffClose, 
            distances.tunlSSA_CsI_oneBD.standoffMid,
            distances.tunlSSA_CsI_oneBD.standoffFar]
standoffName = {0: 'close', 1:'mid', 2:'far'}

tofWindowSettings = tofWindows.csi_oneBD()
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

tof_nBins = tofWindowSettings.nBins
tof_minRange = [tofWindowSettings.minRange['close'], 
                tofWindowSettings.minRange['mid'], 
                tofWindowSettings.minRange['far']]
tof_maxRange = [tofWindowSettings.maxRange['close'], 
                tofWindowSettings.maxRange['mid'], 
                tofWindowSettings.maxRange['far']]
tof_range = []
for i in range(nRuns):
    tof_range.append((tof_minRange[i],tof_maxRange[i]))
tofRunBins = [tof_nBins['close'], 
                tof_nBins['mid'], 
                tof_nBins['far']]


# range of eD expanded for seemingly higher energy oneBD neutron beam
eD_bins = 100
# if quickAndDirty == True:
#     eD_bins = 20
eD_minRange = 200.0
eD_maxRange = 2200.0
eD_range = (eD_minRange, eD_maxRange)
eD_binSize = (eD_maxRange - eD_minRange)/eD_bins
eD_binCenters = np.linspace(eD_minRange + eD_binSize/2,
                            eD_maxRange - eD_binSize/2,
                            eD_bins)


x_bins = 10
# if quickAndDirty == True:
#     x_bins = 5
x_minRange = 0.0
x_maxRange = distances.tunlSSA_CsI_oneBD.cellLength
x_range = (x_minRange,x_maxRange)
x_binSize = (x_maxRange - x_minRange)/x_bins
x_binCenters = np.linspace(x_minRange + x_binSize/2,
                           x_maxRange - x_binSize/2,
                           x_bins)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# SETUP OF SOME SAMPLING PARAMETERS
#
# THESE CAN BE IMPORTANT IN PERFORMANCE
#

nSamples = 200000
if quickAndDirty == True:
    nSamples = 50000
    nDrawsPerEval = nSamples

# parameters for making the fake data...
nEvPerLoop = 10000
data_x = np.repeat(x_binCenters,nEvPerLoop)

#
#
#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


ddnXSinstance = ddnXSinterpolator()
# 
# SET UP THE BEAM TIMING SPREAD
# 
# for one-BD data, binning is 4ns
# spread, based on TF1 fits to gamma peak, is ~4 ns
# to incorporate potential binning errors, maybe 4+4 in quadrature? (this is ~5.65)
# this is perhaps not where binning errors should be accommodated
# anyway, first arg sets timing spread sigma
beamTiming = beamTimingShape.gaussianTiming(2, 4)
zeroDegTimeSpreader = zeroDegreeTimingSpread()

# stopping power model and parameters
stoppingMedia_Z = 1
stoppingMedia_A = 2
stoppingMedia_rho = 8.565e-5 # from red notebook, p 157
incidentIon_charge = 1

# NOTE: this value (19.2) is from PDG
# http://pdg.lbl.gov/2016/AtomicNuclearProperties/HTML/deuterium_gas.html
# it is in ELECTRONVOLTS
# alt ref: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19840013417.pdf
# where this value is 18.2 for H_2 gas
# anyway, the programming in ionStopping assumes keV
# thus the factor of 1e-3
stoppingMedia_meanExcitation = 19.2 * 1e-3 # 

stoppingModelParams = [stoppingMedia_Z, stoppingMedia_A, stoppingMedia_rho,
                       incidentIon_charge, stoppingMedia_meanExcitation]
stoppingModel = ionStopping.simpleBethe( stoppingModelParams )

    
eN_binCenters = getDDneutronEnergy( eD_binCenters )

eD_stoppingApprox_binning = (100,2400,100)

stoppingApprox = ionStopping.betheApprox(stoppingModel, eD_stoppingApprox_binning, x_binCenters)

    
def getTOF(mass, energy, distance):
    """Compute time of flight, in nanoseconds, given\
    mass of particle (in keV/c^2), the particle's energy (in keV),\
    and the distance traveled (in cm).
    Though simple enough to write inline, this will be used often.
    """
    velocity = physics.speedOfLight * np.sqrt(2 * energy / mass)
    tof = distance / velocity
    return tof
    
    

def generateModelData_ode(params, standoffDistance, range_tof, nBins_tof, ddnXSfxn,
                      dedxfxn, beamTimer, nSamples, getPDF=False):
    """
    Generate model data with cross-section weighting applied
    ddnXSfxn is an instance of the ddnXSinterpolator class -
    dedxfxn is a function used to calculate dEdx -
    probably more efficient to these in rather than reinitializing
    one each time
    This is edited to accommodate multiple standoffs being passed 

    bgLevel parameter treats flat background - it is the lambda parameter of a poisson that is sampled from in each TOF bin
    """
    beamE, eLoss, scale, s, scaleFactor, bgLevel = params
    e0mean = 900.0
    dataHist = np.zeros((x_bins, eD_bins))
    
    dedxForODE = lambda x, y: dedxfxn(energy=y,x=x)
    
    nLoops = int(np.ceil(nSamples / nEvPerLoop))
    for loopNum in range(0, nLoops):
        #eZeros = np.random.normal( params[0], params[0]*params[1], nEvPerLoop )
        #eZeros = skewnorm.rvs(a=skew0, loc=e0, scale=e0*sigma0, size=nEvPerLoop)
        eZeros = np.repeat(beamE, nEvPerLoop)
        eZeros -= lognorm.rvs(s=s, loc=eLoss, scale=scale, size=nEvPerLoop)
        checkForBadEs = True
        while checkForBadEs:
            badIdxs = np.where(eZeros <= 0.0)[0]
            nBads = badIdxs.shape[0]
            if nBads == 0:
                checkForBadEs = False
            replacements = np.repeat(beamE, nBads) - lognorm.rvs(s=s, loc=eLoss, scale=scale, size=nBads)
            eZeros[badIdxs] = replacements

#data_eD_matrix = odeint( dedxfxn, eZeros, x_binCenters )
        
        odesolver = ode( dedxForODE ).set_integrator('dopri5').set_initial_value(eZeros)
        for idx, xEvalPoint in enumerate(x_binCenters):
            sol = odesolver.integrate( xEvalPoint )
            #
            # STUFF FOR DEBUGGING
            #
            #print('shape of returned ode solution {}, first 10 entries {}'.format(sol.shape, sol[:10]))
            # data_eD_matrix = odesolver.integrate( x_binCenters )
                #print('shape of returned ode solution {}, first 10 entries {}'.format(data_eD_matrix.shape, data_eD_matrix[:10]))
            # data_eD = data_eD_matrix.flatten('K')
            #
            # END STUFF FOR DEBUGGING
            #
            data_weights = ddnXSfxn.evaluate(sol)
            hist, edEdges = np.histogram( sol, bins=eD_bins, range=(eD_minRange, eD_maxRange), weights=data_weights)
            dataHist[idx,:] += hist
    #
    # DEBUGGING
    #
#    print('length of data_x {} length of data_eD {} length of weights {}'.format(
#          len(data_x), len(data_eD), len(data_weights)))
#     dataHist2d, xedges, yedges = np.histogram2d( data_x, data_eD,
#                                               [x_bins, eD_bins],
#                                               [[x_minRange,x_maxRange],[eD_minRange,eD_maxRange]],
#                                               weights=data_weights)
#     dataHist += dataHist2d # element-wise, in-place addition


            
# #    print('linalg norm value {}'.format(np.linalg.norm(dataHist)))
#     dataHist = dataHist / np.linalg.norm(dataHist)
# #    print('sum of data hist {}'.format(np.sum(dataHist*eD_binSize*x_binSize)))
#     dataHist /= np.sum(dataHist*eD_binSize*x_binSize)
#     plot.matshow(dataHist)
#     plot.show()
    #
    # END DEBUGGING
    #
    e0mean = np.mean(eZeros)
    drawHist2d = (np.rint(dataHist * nSamples)).astype(int)
    tofs = []
    tofWeights = []
    for index, weight in np.ndenumerate( drawHist2d ):
        cellLocation = x_binCenters[index[0]]
        effectiveDenergy = (e0mean + eD_binCenters[index[1]])/2
        tof_d = getTOF( masses.deuteron, effectiveDenergy, cellLocation )
        neutronDistance = (distances.tunlSSA_CsI.cellLength - cellLocation +
                           standoffDistance )
        tof_n = getTOF(masses.neutron, eN_binCenters[index[1]], neutronDistance)
        zeroD_times, zeroD_weights = zeroDegTimeSpreader.getTimesAndWeights( eN_binCenters[index[1]] )
        tofs.append( tof_d + tof_n + zeroD_times )
        tofWeights.append(weight * zeroD_weights)
        # TODO: next line needs adjustment if using OLD NUMPY < 1.6.1 
        # if lower than that, use the 'normed' arg, rather than 'density'
    tofData, tofBinEdges = np.histogram( tofs, bins=nBins_tof, range=range_tof,
                                        weights=tofWeights, density=getPDF)
                                        
    # return step applies scaling and adds poisson-distributed background
    return scaleFactor * beamTimer.applySpreading(tofData) + np.random.poisson(bgLevel, nBins_tof)


# TODO: make better implementation of 0deg transit time
zeroDegSpread_binCenters = np.linspace(0, 24, 7, True)
zeroDegSpread_vals = np.exp(-zeroDegSpread_binCenters/2) /np.sum(np.exp(-zeroDegSpread_binCenters/2))

def generateModelData(params, standoffDistance, range_tof, nBins_tof, ddnXSfxn,
                      stoppingApprox, beamTimer, nSamples, getPDF=False):
    """
    Generate model data with cross-section weighting applied
    ddnXSfxn is an instance of the ddnXSinterpolator class -
    stoppingApprox - instance of betheApprox class
    This is edited to accommodate multiple standoffs being passed 

    bgLevel parameter treats flat background - it is the lambda parameter of a poisson that is sampled from in each TOF bin
    """
    beamE, eLoss, scale, s, scaleFactor, bgLevel = params
    e0mean = 900.0
    dataHist = np.zeros((x_bins, eD_bins))
    
    
    
    nLoops = int(np.ceil(nSamples / nEvPerLoop))
    solutions = np.zeros((nEvPerLoop,x_bins))
    dataHist = np.zeros((x_bins,eD_bins))
    for loopNum in range(0, nLoops):
        #eZeros = np.random.normal( params[0], params[0]*params[1], nEvPerLoop )
        #eZeros = skewnorm.rvs(a=skew0, loc=e0, scale=e0*sigma0, size=nEvPerLoop)
        eZeros = np.repeat(beamE, nEvPerLoop)
        eZeros -= lognorm.rvs(s=s, loc=eLoss, scale=scale, size=nEvPerLoop)
        # checkForBadEs = True
        # while checkForBadEs:
        #     badIdxs = np.where(eZeros <= 0.0)[0]
        #     nBads = badIdxs.shape[0]
        #     if nBads == 0:
        #         checkForBadEs = False
        #     replacements = np.repeat(beamE, nBads) - lognorm.rvs(s=s, loc=eLoss, scale=scale, size=nBads)
        #     eZeros[badIdxs] = replacements

#data_eD_matrix = odeint( dedxfxn, eZeros, x_binCenters )
        #solutions = np.zeros((1,x_bins))
        
        for idx,eZero in enumerate(eZeros):
            sol = stoppingApprox.evalStopped(eZero, x_binCenters)
            #solutions = np.vstack((solutions,sol))
            solutions[idx,:] = sol
        #solutions = solutions[1:,:]

        #dataHist = np.zeros(eD_bins)
        
        for idx,xStepSol in enumerate(solutions.T):
            data_weights = ddnXSfxn.evaluate(xStepSol)
            hist, edEdges = np.histogram(xStepSol, bins=eD_bins, range=(eD_minRange, eD_maxRange), weights=data_weights)
            #dataHist = np.vstack((dataHist,hist))
            dataHist[idx,:] = hist
        #dataHist = dataHist[1:,:]
    #
    # DEBUGGING
    #
#    print('length of data_x {} length of data_eD {} length of weights {}'.format(
#          len(data_x), len(data_eD), len(data_weights)))
#     dataHist2d, xedges, yedges = np.histogram2d( data_x, data_eD,
#                                               [x_bins, eD_bins],
#                                               [[x_minRange,x_maxRange],[eD_minRange,eD_maxRange]],
#                                               weights=data_weights)
#     dataHist += dataHist2d # element-wise, in-place addition


            
# #    print('linalg norm value {}'.format(np.linalg.norm(dataHist)))
#     dataHist = dataHist / np.linalg.norm(dataHist)
# #    print('sum of data hist {}'.format(np.sum(dataHist*eD_binSize*x_binSize)))
#     dataHist /= np.sum(dataHist*eD_binSize*x_binSize)
#     plot.matshow(dataHist)
#     plot.show()
    #
    # END DEBUGGING
    #
    e0mean = np.mean(eZeros)
    drawHist2d = (np.rint(dataHist * nSamples)).astype(int)
    #tofs = []
    #tofWeights = []
    tofs = np.zeros((x_bins,eD_bins))
    tofWeights = np.zeros((x_bins,eD_bins))
    #print('shape of drawHist2d {}\n'.format(drawHist2d.shape))
    for index, weight in np.ndenumerate( drawHist2d ):
        cellLocation = x_binCenters[index[0]]
        effectiveDenergy = (e0mean + eD_binCenters[index[1]])/2
        tof_d = getTOF( masses.deuteron, effectiveDenergy, cellLocation )
        neutronDistance = (distances.tunlSSA_CsI.cellLength - cellLocation +
                           standoffDistance )
        tof_n = getTOF(masses.neutron, eN_binCenters[index[1]], neutronDistance)
        # TODO: reimplement zero degree transit spread
        #zeroD_times, zeroD_weights = zeroDegTimeSpreader.getTimesAndWeights( eN_binCenters[index[1]] )
        #tofs.append( tof_d + tof_n + zeroD_times )
        #tofWeights.append(weight * zeroD_weights)
        tofs[index] = tof_d + tof_n #+ zeroD_times
        tofWeights[index] = weight #* zeroD_weights
    #!!!!!!!!
    # debugging
    #print('shape of tofs and tofWeights: {} {}\n'.format(len(tofs), len(tofWeights)))
        # TODO: next line needs adjustment if using OLD NUMPY < 1.6.1 
        # if lower than that, use the 'normed' arg, rather than 'density'
    tofData, tofBinEdges = np.histogram( tofs.flatten(), bins=nBins_tof, range=range_tof,
                                        weights=tofWeights.flatten(), density=getPDF)
                                        
    # spread with expo modeling transit time across 0 degree
    tofData = np.convolve(tofData, zeroDegSpread_vals, 'full')[:-len(zeroDegSpread_binCenters)+1]
    # return step applies scaling and adds poisson-distributed background
    return scaleFactor * beamTimer.applySpreading(tofData) + np.random.poisson(bgLevel, nBins_tof)

    

def genModelData_lowMem(params, standoffDistance, range_tof, nBins_tof, ddnXSfxn,
                      dedxfxn, beamTimer, nSamples, getPDF=False):
    """
    Generate model data with cross-section weighting applied
    
    this iterates more simply, trying to avoid memory mishaps
    """
    beamE, eLoss, scale, s, scaleFactor = params
    dataHist = np.zeros((x_bins, eD_bins))
    for sampleNum in range(nSamples):
#        if sampleNum % 1000 == 0:
#            print('on sample {0}'.format(sampleNum))
        #eZeros = np.random.normal( params[0], params[0]*params[1], nEvPerLoop )
        #eZeros = skewnorm.rvs(a=skew0, loc=e0, scale=e0*sigma0, size=nEvPerLoop)
        eZero = beamE - lognorm.rvs(s=s, loc=eLoss, scale=scale, size=1)[0]
        checkForBadEs = False
        if eZero <= 0:
            checkForBadEs = True
        while checkForBadEs:
            eZero = beamE - lognorm.rvs(s=s, loc=eLoss, scale=scale, size=1)[0]
            if eZero <= 0:
                checkForBadEs = True

        data_eD_matrix = odeint( dedxfxn, eZero, x_binCenters )
        data_eD = data_eD_matrix.flatten()
        data_weights = ddnXSfxn.evaluate(data_eD)
#        print('length of data_x {} length of data_eD {} length of weights {}'.format(
#              len(data_x), len(data_eD), len(data_weights)))
        dataHist2d, xedges, yedges = np.histogram2d( x_binCenters, data_eD,
                                                [x_bins, eD_bins],
                                                [[x_minRange,x_maxRange],[eD_minRange,eD_maxRange]],
                                                weights=data_weights)
        dataHist += dataHist2d # element-wise, in-place addition
        
    e0mean = np.mean(beamE - lognorm.rvs(s=s, loc=eLoss, scale=scale, size=1000))       
#    print('linalg norm value {}'.format(np.linalg.norm(dataHist)))
#    dataHist = dataHist / np.linalg.norm(dataHist)
#    print('sum of data hist {}'.format(np.sum(dataHist*eD_binSize*x_binSize)))
    dataHist /= np.sum(dataHist*eD_binSize*x_binSize)
#    plot.matshow(dataHist)
#    plot.show()
    drawHist2d = (np.rint(dataHist * nSamples)).astype(int)
    tofs = []
    tofWeights = []
    for index, weight in np.ndenumerate( drawHist2d ):
        cellLocation = x_binCenters[index[0]]
        effectiveDenergy = (e0mean + eD_binCenters[index[1]])/2
        tof_d = getTOF( masses.deuteron, effectiveDenergy, cellLocation )
        neutronDistance = (distances.tunlSSA_CsI.cellLength - cellLocation +
                           standoffDistance )
        tof_n = getTOF(masses.neutron, eN_binCenters[index[1]], neutronDistance)
        zeroD_times, zeroD_weights = zeroDegTimeSpreader.getTimesAndWeights( eN_binCenters[index[1]] )
        tofs.append( tof_d + tof_n + zeroD_times )
        tofWeights.append(weight * zeroD_weights)
        # TODO: next line needs adjustment if using OLD NUMPY < 1.6.1 
        # if lower than that, use the 'normed' arg, rather than 'density'
    tofData, tofBinEdges = np.histogram( tofs, bins=nBins_tof, range=range_tof,
                                        weights=tofWeights, density=getPDF)
    return scaleFactor * beamTimer.applySpreading(tofData) 


def lnlikeHelp(evalData, observables):
    """
    helper function for use in lnlike function and its possible parallelization
    handles convolution of beam-timing characteristics with fake data
    then actually does the likelihood eval and returns likelihood value
    """
    logEvalHist = np.log(evalData)
    zeroObservedIndices = np.where(observables == 0)[0]
    for idx in zeroObservedIndices:
        if logEvalHist[idx] == -inf:
            logEvalHist[zeroObservedIndices] = 0

    return np.dot(logEvalHist,observables) # returns loglike value


def lnlike(params, observables, standoffDist, range_tof, nBins_tof, 
           nDraws=nSamples):
    """
    Evaluate the log likelihood using xs-weighting
    """        
    #e0, sigma0 = params
    evalData = generateModelData(params, standoffDist, range_tof, nBins_tof,
                                    ddnXSinstance, stoppingApprox,
                                    beamTiming, nDraws, True)
    binLikelihoods = []
    for binNum in range(len(observables)):
        if observables[binNum] == 0:
            observables[binNum] = 1
        if evalData[binNum] == 0:
            evalData[binNum] = 1
# these nexxt two lines are a poor/old man's poisson.logpmf()
# note that np.log(poisson.pmf()) is NOT THE SAME!
        poiLogpmf = -observables[binNum] - special.gammaln(int(evalData[binNum])+1)
        if evalData[binNum] > 0:
            poiLogpmf += evalData[binNum]*np.log(observables[binNum])
        binLikelihoods.append(observables[binNum] * poiLogpmf)
#        binLikelihoods.append(norm.logpdf(evalData[binNum], 
#                                          observables[binNum], 
#                                            observables[binNum] * 0.10))
#        binLikelihoods.append(norm.logpdf(observables[binNum],
#                                          evalData[binNum],
#                                            evalData[binNum]*0.15))
#    print('bin likelihoods {}'.format(binLikelihoods))
#    print('returning overall likelihood of {}'.format(np.sum(binLikelihoods)))
    return np.sum(binLikelihoods)
    
    
def compoundLnlike(params, observables, standoffDists, tofRanges, tofBinnings, 
                   nDraws=nSamples):
    """Compute the joint likelihood of the model with each of the runs at different standoffs"""
    paramSets = [[params[0], params[1], params[2], params[3], scale, bgLevel] for scale, bgLevel in zip(params[4:-nRuns], params[-nRuns:])]
    loglike = [lnlike(paramSet, obsSet, standoff, tofrange, tofbin, nDraws) for
               paramSet, obsSet, standoff, tofrange, tofbin in 
               zip(paramSets, observables, standoffDists, tofRanges, 
                   tofBinnings)]
    return np.sum(loglike)
    
    
    
# PARAMETER BOUNDARIES
min_beamE, max_beamE = 1500.0, 2400.0 # CONFER WITH TANDEM LOGS
min_eLoss, max_eLoss = 400.0,1200.0
min_scale, max_scale = 40.0, 500.0
min_s, max_s = 0.05, 2.0
paramRanges = []
paramRanges.append((min_beamE, max_beamE))
paramRanges.append((min_eLoss, max_eLoss))
paramRanges.append((min_scale, max_scale))
paramRanges.append((min_s, max_s))
for i in range(nRuns):
    paramRanges.append( (1e3, 1.0e8) ) # scale factors are all allowed to go between 0 and 30000  (??) for now
for i in range(nRuns):
    paramRanges.append( (0.0, 1e3) ) # background levels
    
    

def lnprior(theta):
    # run through list of params and if any are outside of allowed range, immediately return -inf
    for idx, paramVal in enumerate(theta):
        if paramVal < paramRanges[idx][0] or paramVal > paramRanges[idx][1]:
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # DEBUGGING
            #
            # print('param {} has value {} - OUT OF RANGE\n'.format(idx, paramVal))
            #
            # DEBUGGING
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            return -inf
    return 0
    
def lnprob(theta, observables, standoffDists, tofRanges, nTOFbins, nDraws=nDrawsPerEval):
    """Evaluate the log probability
    theta is a list of the model parameters
        E0, E0_sigma, scaleFactors0-4
    observables is, in this case, a list of histograms of TOF values
    
    """
    prior = lnprior(theta)
#    for param in theta:
#        print('param value {}'.format(param))
#    print('prior has value {}'.format(prior))
    if not np.isfinite(prior):
        return -inf
    loglike = compoundLnlike(theta, observables, standoffDists, tofRanges, 
                             nTOFbins, nDraws)
#    gc.collect()
#    print('loglike value {}'.format(loglike))
    logprob = prior + loglike
#    print('logprob has value {}'.format(logprob))
    if isnan(logprob):
        print('\n\n\n\nWARNING\nlogprob evaluated to NaN\nDump of observables and then parameters used for evaluation follow\n\n\n')
        print(observables)
        print('\n\nPARAMETERS\n\n')
        print(theta)
        return -inf
    return logprob
   

    
    
def checkLikelihoodEval(observables, evalData):
    """Verbosely calculate the binned likelhood for a set of observables and model data"""
    nBins = len(observables)
    binLikelihoods = []
    for binNum in range(nBins):
        if observables[binNum] == 0:
            print('observable has 0 data in bin {0}, setting to 1'.format(binNum))
            observables[binNum] = 1
        if evalData[binNum] == 0:
            print('model has 0 data in bin {0}, setting to 1'.format(binNum))
            evalData[binNum]= 1
        binlike = observables[binNum] * norm.logpdf(evalData[binNum], 
                                          observables[binNum], 
                                            observables[binNum] * 0.10)
        binlike += norm.logpdf(observables[binNum],
                                          evalData[binNum],
                                            evalData[binNum]*0.15)
        binLikelihoods.append(observables[binNum]*norm.logpdf(evalData[binNum], 
                                          observables[binNum], 
                                            observables[binNum] * 0.10))
        binLikelihoods.append(norm.logpdf(observables[binNum],
                                          evalData[binNum],
                                            evalData[binNum]*0.15))
        print('bin {0} has likelihood {1}'.format(binNum, binlike))
    
    print('total likelihood is {0}'.format(np.sum(binLikelihoods)))
    
    simpleIndices = np.arange(nBins)
    fig, (axOverlay, axResid) = plot.subplots(2)
    axOverlay.scatter(simpleIndices, observables, color='green')
    axOverlay.scatter(simpleIndices, evalData, color='red')
    axOverlay.set_ylabel('Counts')
    axResid.scatter(simpleIndices, observables - evalData )
    axResid.set_ylabel('Residual')
    
        
    
    plot.draw()
    #plot.show()

    
    

    
    



# get the data from file
tofData = readMultiStandoffTOFdata(inputDataFilename, 3)


binEdges = tofData[:,0]

# observedTOF = tofData[:,runNumber+1][(binEdges >= tof_minRange) & (binEdges < tof_maxRange)]
# observedTOFbinEdges = tofData[:,0][(binEdges>=tof_minRange)&(binEdges<tof_maxRange)]

observedTOF = []
observedTOFbinEdges=[]
for i in range(nRuns):
    observedTOF.append(tofData[:,i+1][(binEdges >= tof_minRange[i]) & (binEdges < tof_maxRange[i])])
    observedTOFbinEdges.append(tofData[:,0][(binEdges>=tof_minRange[i])&(binEdges<tof_maxRange[i])])


beamE_guess = 2000.0 # initial deuteron energy, in keV
eLoss_guess = 750.0 # width of initial deuteron energy spread
scale_guess = 180.0
s_guess = 1.0

paramGuesses = [beamE_guess, eLoss_guess, scale_guess, s_guess]
#badGuesses = [e0_bad, sigma0_bad, skew_bad]
scaleFactor_guesses = []
for i in range(nRuns):
    scaleFactor_guesses.append(5 * np.sum(observedTOF[i]))
    paramGuesses.append(np.sum(observedTOF[i]))
    #badGuesses.append(np.sum(observedTOF[i]))

bgLevel_guesses = []
for i in range(nRuns):
    bgLevel_guesses.append(10)
    paramGuesses.append(10)




if not useMPI:
    if debugging and doPlotting:
        #nSamples = 5000
        fakeData1 = generateModelData([beamE_guess, eLoss_guess, scale_guess, s_guess, 5000, bgLevel_guesses[0]], 
                                     standoffs[0], tof_range[0], tofRunBins[0], 
                                        ddnXSinstance, stoppingApprox, beamTiming,
                                        nEvPerLoop, getPDF=True)
        tofbins = np.linspace(tof_minRange[0], tof_maxRange[0], tofRunBins[0])
        plot.figure()
        plot.plot(tofbins, fakeData1)
        plot.draw()
        #plot.show()
            
    
    # generate fake data
    
    
    if not batchMode:
        fakeData = [generateModelData([beamE_guess, eLoss_guess, scale_guess, s_guess, sfGuess, bgGuess],
                                      standoff, tofrange, tofbins, 
                                      ddnXSinstance, stoppingApprox, beamTiming,
                                      nSamples, getPDF=True) for 
                                      sfGuess, bgGuess, standoff, tofrange, tofbins in 
                                      zip(scaleFactor_guesses, bgLevel_guesses, standoffs, tof_range,
                                          tofRunBins)]

        if doPlotting:
            # plot the TOF
           
            tofbins = []
            for idx in range(len(tof_minRange)):
                tofbins.append(np.linspace(tof_minRange[idx], tof_maxRange[idx], tofRunBins[idx]))
            plot.figure()
            plot.subplot(211)
            for i in range(len(tof_minRange)):
                plot.scatter(tofbins[i], observedTOF[i], color=runColors[i])
            plot.xlim(min(tof_minRange), max(tof_maxRange))
            plot.ylabel('Experimental observed counts')
            plot.title('Observed data and fake data for two parameter sets')
            plot.subplot(212)
            for i in range(len(tof_minRange)):
                plot.scatter(tofbins[i], fakeData[i], color=runColors[i], marker='d')
                #plot.scatter(tofbins[i], fakeDataOff[i], color=runColors[i], marker='+')
            plot.ylabel('counts')
            plot.xlabel('TOF (ns)')
            plot.xlim(min(tof_minRange),max(tof_maxRange))
            
            plot.draw()
            
            #plot.show()

            fig, ax = plot.subplots(figsize=(8.5,5.25))
            tofbins = []
            for idx in range(len(tof_minRange)):
                tofbins.append(np.linspace(tof_minRange[idx], tof_maxRange[idx], tofRunBins[idx]))
            ax.set_xlim(min(tof_minRange), max(tof_maxRange))
            for i in range(len(tof_minRange)):
                plot.scatter(tofbins[i], observedTOF[i],color=runColors[i], label='Exp. data')
                plot.scatter(tofbins[i], fakeData[i], edgecolors=runColors[i], facecolors='none', label='Fake data')
            ax.legend( loc='best', frameon=False, ncol=2, numpoints=1)

            ax.set_xlabel('TOF (ns)')

            plot.draw()
            #plot.show()

        if debugging:
            nll = lambda *args: -compoundLnlike(*args)
            
            
            testNLL = nll(paramGuesses, observedTOF, standoffs, tof_range, tofRunBins)
            print('test NLL has value {0}'.format(testNLL))
            
            testProb = lnprob(paramGuesses, observedTOF, standoffs, tof_range, tofRunBins)
            print('got test lnprob {0}'.format(testProb))


if quitEarly:
    plot.show()
    quit()

#
# HERE'S WHERE THE MCMC SAMPLING LIVES
#
if outputPrefix == '':
    burninChainFilename = 'burninchain.dat'
    mainChainFilename = 'mainchain.dat'
else:
    burninChainFilename = outputPrefix + '_burninchain.dat'
    mainChainFilename = outputPrefix + '_mainchain.dat'

nDim, nWalkers = len(paramGuesses), parsedArgs.nWalkers
if debugging:
    nWalkers = 2 * nDim
    if parsedArgs.nWalkers != 256:
        nWalkers = parsedArgs.nWalkers

p0agitators = [10, 50, 20, 0.05]
for guess in paramGuesses[4:]:
    p0agitators.append(guess * 0.15)

p0 = [paramGuesses + p0agitators*np.random.randn(nDim) for i in range(nWalkers)]

# NOTE: i am not copying over the MPI-enabled stuff right now
sampler = emcee.EnsembleSampler(nWalkers, nDim, lnprob, 
                                    kwargs={'observables': observedTOF,
                                            'standoffDists': standoffs,
                                            'tofRanges': tof_range,
                                            'nTOFbins': tofRunBins},
                                    threads=nThreads)
fout = open(burninChainFilename,'w')
fout.close()

if debugging:
    burninSteps = 10
    if parsedArgs.nBurninSteps != 400:
        burninSteps = parsedArgs.nBurninSteps

print('\n\n\nRUNNING BURN IN WITH {0} STEPS\n\n\n'.format(burninSteps))

for i,samplerOut in enumerate(sampler.sample(p0, iterations=burninSteps)):
    if not useMPI or processPool.is_master():
        burninPos, burninProb, burninRstate = samplerOut
        print('running burn-in step {0} of {1}...'.format(i, burninSteps))
        fout = open(burninChainFilename, "a")
        for k in range(burninPos.shape[0]):
            fout.write("{0} {1} {2}\n".format(k, burninPos[k], burninProb[k]))
        fout.close()


e0_only = False

if doPlotting:
    # save an image of the burn in sampling
    if not e0_only:
        plot.figure()
        plot.subplot(411)
        plot.plot(sampler.chain[:,:,0].T,'-',color='k',alpha=0.2)
        plot.ylabel(r'$E_0$ (keV)')
        plot.subplot(412)
        plot.plot(sampler.chain[:,:,1].T,'-',color='k',alpha=0.2)
        plot.ylabel(r'loc (keV)')
        plot.subplot(413)
        plot.plot(sampler.chain[:,:,2].T,'-',color='k',alpha=0.2)
        plot.ylabel(r'scale')
        plot.subplot(414)
        plot.plot(sampler.chain[:,:,3].T,'-', color='k', alpha=0.2)
        plot.ylabel(r's')
        plot.xlabel('Burn-in step')
    else:
        plot.figure()
        plot.plot( sampler.chain[:,:,0].T, '-', color='k', alpha=0.2)
        plot.ylabel(r'$E_0$ (keV)')
        plot.xlabel('Step')
    plot.savefig(outputPrefix + 'emceeBurninSampleChainsOut.png',dpi=300)
    plot.draw()


if not useMPI or processPool.is_master():
    fout = open(mainChainFilename,'w')
    fout.close()

sampler.reset()

if debugging:
    if parsedArgs.nMainSteps != 100:
        mcIterations = 10
    else:
        mcIterations = parsedArgs.nMainSteps
for i,samplerResult in enumerate(sampler.sample(burninPos, lnprob0=burninProb, rstate0=burninRstate, iterations=mcIterations)):
    #if (i+1)%2 == 0:
    #    print("{0:5.1%}".format(float(i)/mcIterations))
    print('running step {0} of {1} in main chain'.format(i, mcIterations))
    fout = open(mainChainFilename,'a')
    pos=samplerResult[0]
    prob = samplerResult[1]
    for k in range(pos.shape[0]):
        fout.write("{0} {1} {2}\n".format(k, pos[k], prob[k]))
    fout.close()

    
if useMPI:
    processPool.close()
    


samples = sampler.chain[:,:,:].reshape((-1,nDim))

if not e0_only:
    # Compute the quantiles.
    # this comes from https://github.com/dfm/emcee/blob/master/examples/line.py
    quartileResults = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                          zip(*np.percentile(samples, [16, 50, 84], axis=0))))

    ed_mcmc, loc_mcmc, scale_mcmc, s_mcmc = quartileResults[:4]
    print("""MCMC result:
        E_D initial = {0[0]} +{0[1]} -{0[2]}
        loc = {1[0]} +{1[1]} -{1[2]}
        scale = {2[0]} + {2[1]} - {2[2]}
        s = {3[0]} + {3[1]} - {3[2]}
        """.format(ed_mcmc, loc_mcmc, scale_mcmc, s_mcmc))
    
    
#    import corner as corn
#    cornerFig = corn.corner(samples,labels=["$E_0$","$\sigma_0$, skew"],
#                            quantiles=[0.16,0.5,0.84], show_titles=True,
#                            title_kwargs={'fontsize': 12})
#    cornerFig.savefig('emceeRunCornerOut.png',dpi=300)
else:
    getQuartilesFromPercentiles = lambda v:(v[1], v[2]-v[1],v[1]-v[0])

    e0_mcmc = getQuartilesFromPercentiles(np.percentile(samples, [16,50,84], axis=0).T[0,:])
    print("""MCMC result:
        E0 = {0[0]} +{0[1]} -{0[2]}
        """.format(e0_mcmc))
        
        
if doPlotting:
    if not e0_only:
        plot.figure()
        plot.subplot(411)
        plot.plot(sampler.chain[:,:,0].T,'-',color='k',alpha=0.2)
        plot.ylabel(r'$E_0$ (keV)')
        plot.subplot(412)
        plot.plot(sampler.chain[:,:,1].T,'-',color='k',alpha=0.2)
        plot.ylabel(r'loc (keV)')
        plot.subplot(413)
        plot.plot(sampler.chain[:,:,2].T,'-',color='k',alpha=0.2)
        plot.ylabel(r'scale')
        plot.subplot(414)
        plot.plot(sampler.chain[:,:,3].T,'-', color='k', alpha=0.2)
        plot.ylabel(r's')
        plot.xlabel('Step')
    else:
        plot.figure()
        plot.plot(sampler.chain[:,:,0].T,'-',color='k',alpha=0.2)
        plot.ylabel(r'$E_0$ (keV)')
        plot.xlabel('Step')
    plot.savefig(outputPrefix + 'emceeRunSampleChainsOut.png',dpi=300)
    plot.draw()    
    plot.show()