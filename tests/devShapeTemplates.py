#
#

from __future__ import print_function
import numpy as np
from numpy import inf
import scipy.optimize as optimize
from scipy.integrate import odeint
from scipy.stats import (poisson, norm)
import matplotlib.pyplot as plot
import emcee
import csv as csvlib
import argparse
from numbers import Number
from constants.constants import (masses, distances, physics, tofWindows)
from utilities.utilities import (beamTimingShape, ddnXSinterpolator, 
                                 getDDneutronEnergy)
from utilities.utilities import readMultiStandoffTOFdata
from utilities.ionStopping import ionStopping
from math import isnan


#
# SHAPE OF PARAMS PASSED TO PROB
# [0:2] - SCALING FACTORS FOR EACH RUN, MINUS 1st, LIST of 3 FLOATS
# [4:] - COEFFICIENT LIST, LIST OF 25 FLOATS
# 
# SHAPE OF PARAMS PASSED TO LIKELIHOOD
# [0] - SCALING FACTOR FOR SPECIFIC RUN, FLOAT
# [1:] - COEFFICIENT LIST 25 FLOATS
#



argParser = argparse.ArgumentParser()
argParser.add_argument('-filename',type=str)
argParser.add_argument('-templateFile', type=str)
argParser.add_argument('-loadTemplates',type=str, default='f')
parsedArgs = argParser.parse_args()
filename = parsedArgs.filename
templateFilename = parsedArgs.templateFile
loadTemplates = False
if parsedArgs.loadTemplates == 't':
    loadTemplates = True
print(loadTemplates)

standoffs = [distances.tunlSSA_CsI.standoffMid, 
             distances.tunlSSA_CsI.standoffClose,
             distances.tunlSSA_CsI.standoffClose,
             distances.tunlSSA_CsI.standoffFar]
             
tofWindowSettings = tofWindows()             
tof_nBins = tofWindowSettings.nBins
tof_minRange = [tofWindowSettings.minRange['mid'], 
                tofWindowSettings.minRange['close'], 
                tofWindowSettings.minRange['close'],
                tofWindowSettings.minRange['far'] ]
tof_maxRange = [tofWindowSettings.maxRange['mid'], 
                tofWindowSettings.maxRange['close'], 
                tofWindowSettings.maxRange['close'],
                tofWindowSettings.maxRange['far'] ]
tof_range = []
for i in range(4):
    tof_range.append((tof_minRange[i],tof_maxRange[i]))
tofRunBins = [tof_nBins['mid'], tof_nBins['close'], 
           tof_nBins['close'], tof_nBins['far']]
           
           
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
nEvPerLoop = 200000
data_x = np.repeat(x_binCenters,nEvPerLoop)
           


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




def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def getBinCenters(eRange, eBins):
    binWidth = (eRange[1]-eRange[0])/eBins
    binCenters = np.linspace(eRange[0] + 1/2*binWidth, eRange[1] - 1/2*binWidth, eBins)
    return binCenters
    
    
def getGuessParams(ed_range, ed_bins, useLog=True):
    # get the centers of the bins
    binCenters = getBinCenters(ed_range, ed_bins)
    binWidth = (binCenters[1]-binCenters[0])/2.0
    guesses = 30000*norm.pdf(binCenters, loc=800, scale=75)*binWidth*5
    if useLog:
        return np.log(guesses)
    return guesses
    
def getGuessParams_square(ed_range, ed_bins, useLog=False):
    # get the centers of the bins
    binCenters = getBinCenters(ed_range, ed_bins)
    guesses = np.zeros(ed_bins)
    for idx in range(ed_bins):
        if binCenters[idx] >= 700 and binCenters[idx] <= 900:
            guesses[idx] = 1
    scale = 20000 / sum(guesses) * 8
    if useLog:
        return np.log(guesses*scale)
    return guesses*scale

def getGuessParams_model(ed_range, ed_bins, useLog = False):
    """This is kind of a weird kernel estimate of the model of deuteron spectrum
    Additive combination of several gaussians"""
    binCenters = getBinCenters(ed_range, ed_bins)
    binWidth = (binCenters[1]-binCenters[0])/2.0
    guesses = 8*(37500*norm.pdf(binCenters, loc=820, scale=75)*binWidth + 20000 * norm.pdf(binCenters, loc=730, scale=125)*binWidth)
    return guesses

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
    
    

def generateModelData(params, standoffDistance, nBins_tof, range_tof, ddnXSfxn, dedxfxn,
                      nSamples, getPDF=False):
    """
    Generate model data with cross-section weighting applied
    ddnXSfxn is an instance of the ddnXSinterpolator class -
    dedxfxn is a function used to calculate dEdx -
    probably more efficient to these in rather than reinitializing
    one each time
    This is edited to accommodate multiple standoffs being passed 
    """
    e0 = params[0]
    dataHist = np.zeros((x_bins, eD_bins))
    nLoops = int(nSamples / nEvPerLoop)
    for loopNum in range(0, nLoops):
#        eZeros = np.repeat(params, nEvPerLoop)
        eZeros = np.random.uniform(e0-12.5, e0+12.5, nEvPerLoop)
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
    tofData, tofBinEdges = np.histogram( tofs, bins=nBins_tof, range=range_tof,
                                    weights=tofWeights, density=getPDF)
    
    return beamTiming.applySpreading(tofData)
    
nTemplates_eD = 80
templateRange_min_eD, templateRange_max_eD = 400, 1200
templateRange_eD = (templateRange_min_eD, templateRange_max_eD)
templateStepSize = (templateRange_max_eD - templateRange_min_eD)/nTemplates_eD
templateGenVals_eD = np.linspace(templateRange_min_eD+templateStepSize/2, templateRange_max_eD - templateStepSize/2, 
                                 nTemplates_eD, endpoint=True)


def buildModelTOF(coeffs, templates):
    #print(type(templates[0]))
    #print(templates[0].shape)
    modelTOF = np.zeros(len(templates[0]))
    scaleFactor = coeffs[0]
    #coeffArray = np.array(coeffs[1:])
    for idx, coeff in enumerate(coeffs[1:]):
#        print(modelTOF)
#        print(coeff)
#        print(templates[idx])
        modelTOF = modelTOF + coeff * templates[idx]
    modelTOF = modelTOF * scaleFactor
    return modelTOF
    
    
    
def lnlike_wide(params, observables, templates):
    """use a very-wide gaussian error on observed data to try to get fit close"""
    logParams = []
#    for val in params[3:]:
#        logParams.append(np.exp(val))
    for idx, val in enumerate(params):
        if idx < 3:
            logParams.append(val)
        else:
            logParams.append(np.exp(val))
    modelTOF = buildModelTOF(params, templates)
    if not np.isfinite(np.sum(modelTOF)):
        return -np.inf
    likes = []
    for i in range(len(observables)):
        if observables[i] == 0:
            observables[i] = 1
        if modelTOF[i] == 0:
            modelTOF[i] = 1
        #likes.append(np.log(poisson.pmf(int(modelTOF[i]), observables[i])))
        likes.append(norm.logpdf(modelTOF[i], observables[i], observables[i] * 0.07))
        likes.append(norm.logpdf(observables[i], modelTOF[i], modelTOF[i] * 0.15))
    return np.sum(likes)
    
    
def lnlike(params, observables, templates):
    modelTOF = buildModelTOF(params, templates)
    if not np.isfinite(np.sum(modelTOF)):
        return -np.inf
    obsErr = np.sqrt(observables)
    inverse_sigma2 = 1.0/(obsErr**2 + modelTOF**2*np.exp(2*np.log(2)))
    sigma = 0.05 * observables
    sigma_model = 0.05 * modelTOF
    for idx,obs in enumerate(observables):
        if obs < 500:
            sigma[idx] = np.fabs(np.sqrt(obs)/obs) * obs
            if obs == 0:
                sigma[idx] = 1.0
        if modelTOF[idx] < 500:
            sigma_model[idx] = np.fabs(np.sqrt(modelTOF[idx])/modelTOF[idx]) * modelTOF[idx]
            if modelTOF[idx] == 0:
                sigma_model[idx] = 1.0
    pointlikes=[]
    for i in range(len(observables)):
        pointlike = (-0.5 * np.log(2 * np.pi * sigma[i]**2) - (observables[i] - modelTOF[i])**2/(2*sigma[i]**2) -
                     0.5 * np.log(2 * np.pi * sigma_model[i]**2) - (observables[i] - modelTOF[i])**2/(2*sigma_model[i]**2))
        print('observable {} model {} sigma {} likelihood {}'.format(observables[i], modelTOF[i], sigma[i], pointlike))
        pointlikes.append( pointlike )
#    likelihood = -0.5*(np.sum(np.log(2 * np.pi * sigma**2) - (observables[i] - modelTOF[i])**2/(2*sigma**2)))
    for i in range(len(observables)):
        if observables[i] == 0:
            observables[i] = 1
        #poilike = np.log(poisson.pmf(int(modelTOF[i]), observables[i]))
        poilike = norm.logpdf(modelTOF[i], observables[i], observables[i] * 0.5)
        print('poisson likelihood {}'.format(poilike))
    likelihood = np.sum(pointlikes)
    tempArr = np.array(observables)
    obsArr = tempArr.astype('int')
    tempArr = np.array(modelTOF)
    modArr = tempArr.astype('int')
    likelihood = np.sum(poisson.pmf(modArr, obsArr))
    print('found total log likelihood for set of {}\n'.format(likelihood))
    return likelihood
    
def compoundLnlike(theta, observables, standoffs, tofbinnings, 
                   tofranges, templates):
    args = [1]
    for coeff in theta[3:]:
        args.append(coeff)
    loglike = lnlike_wide(args, observables[0], templates[0])
    for idx, scale in enumerate(theta[:3]):
        args = [scale]
        for coeff in theta[3:]:
            args.append(coeff)
        loglike = loglike + lnlike_wide( args, observables[idx+1], templates[idx+1])
    return loglike
    
    
scaleLims = [(0.8, 2.0),(0.25, 1.0),(1.3, 1.9)]
def lnprior(theta):
    scaleFactors = theta[:3]
    # check the overall scaling factors
    for idx,scaleFactor in enumerate(scaleFactors):
        if scaleFactor > scaleLims[idx][1] or scaleFactor < scaleLims[idx][0]:
#            print('scale factor tripped inf')
#            print(theta)
            return -inf
    # check the coefficients
    for cv in theta[3:]:
        if cv < 0.0 or cv > 25000:
#        if cv < -20 or cv > 11: # use this for varying the LOG of the coeffs
#            print('coeff tripped inf')
#            print(theta)
            return -inf
    return 0.0
    
def lnprob(theta, observables, standoffs, tofbinnings, tofranges, templates):
    prior = lnprior(theta)
    loglike = compoundLnlike(theta, observables, standoffs, 
                                 tofbinnings, tofranges, templates)
    prob = prior + loglike
    if isnan(prob):
#        print('\n\n\nWARNING\nlnprob found to be NaN')
#        print('prior value {}\nloglike value {}'.format(prior, loglike))
#        print('dumping parameters..\n')
#        print(theta)
#        print('\nreturning -inf...')
        return -inf
    return prior + loglike

    

    
def plotTemplates(eRange, eBins, templates, tofRange, tofBins):
    # make a fig for every 100 keV
    binCenters = getBinCenters(eRange, eBins)
    tofBinCenters = getBinCenters( tofRange, tofBins)
    hundreds = int(np.ceil((eRange[1]-eRange[0])/100))
    rows = int(np.ceil(hundreds/2))
    fig, axes = plot.subplots(rows,2)
    print('got {} axes for {} sets of 100 in {} rows'.format(len(axes), hundreds, rows))
    for idx, template in enumerate(templates):
        axN = int((binCenters[idx]-eRange[0])/200) # dont need to floor it, thats the default
        if (binCenters[idx]-eRange[0]) % 200 >= 100:
            axM = 1
        else:
            axM = 0
#        print(len(binCenters))
#        print(len(template))
        axes[axN][axM].scatter(tofBinCenters, template)
    #plot.draw()
    plot.savefig('templateTOFs_midStandoff.png',dpi=400)
    
    
shapeTemplates = []

nSamples = 200000

if not loadTemplates:
    print('Generating templates...')
    # generate the templates
    templFile = open(templateFilename,'w')
    csvWriter = csvlib.writer(templFile)
    for runIndex, standoff in enumerate(standoffs):
        # make templates for each prototype energy
        standoffTemplates = []
        for energyIdx, energy in enumerate(templateGenVals_eD):
            # make templates at each standoff
            print('generating template for standoff {} of energy {}, or {} keV'.format(runIndex, energyIdx, energy))
            model = generateModelData([energy], standoff, tofRunBins[runIndex], tof_range[runIndex], 
                                                       ddnXSinstance, stoppingModel.dEdx, nSamples, True)
            standoffTemplates.append(model)
            csvWriter.writerow(model)
            #templFile = open(templateFilename, 'a')
            #templFile.write(repr(model))
            #templFile.write('\n')
            #templFile.close()
        shapeTemplates.append(standoffTemplates)
    
    templFile.close()
    
if loadTemplates:
    templFile = open(templateFilename, 'r')
    csvReader = csvlib.reader(templFile)
    for runIndex, standoff in enumerate(standoffs):
        standoffTemplates = []
        for energyIdx, energy in enumerate(templateGenVals_eD):
            #print('reading template for standoff {} of energy {}'.format(runIndex, energyIdx))
            line = next(csvReader)
            modelList = []
            for entry in line:
                modelList.append(float(entry))
            model = np.array(modelList)
            standoffTemplates.append(model)
        shapeTemplates.append(standoffTemplates)
        
print('length of two dimensions of template collection: {}, {}'.format(len(shapeTemplates), len(shapeTemplates[0])))

#plotTemplates(templateRange_eD, nTemplates_eD, shapeTemplates[0], tof_range[0], tofRunBins[0])

# get the EXPERIMENTAL data from file
tofData = readMultiStandoffTOFdata(filename)


binEdges = tofData[:,0]

#observedTOF, observed_bin_edges = np.histogram(fakeData[:,3],
#                                               tof_nBins, tof_range)
observedTOF = []
observedTOFbinEdges=[]
for i in range(4):
    observedTOF.append(tofData[:,i+1][(binEdges >= tof_minRange[i]) & (binEdges < tof_maxRange[i])])
    observedTOFbinEdges.append(tofData[:,0][(binEdges>=tof_minRange[i])&(binEdges<tof_maxRange[i])])

coefficients = []
for i in range(nTemplates_eD + 3):
    coefficients.append(0.001)
# we can make decent guesses on the 1st 3 coeffs based on # of counts in TOF spectra
# relative to run 0
tofCounts = []
for runSpec in observedTOF:
    tofCounts.append(np.sum(runSpec))
for i in range(1,4):
    coefficients[i-1] = float(tofCounts[i]/tofCounts[0])

#initialIndicesForGuess= []
#for idx,entry in enumerate(templateGenVals_eD):
#    if entry >= 700 and entry <= 900:
#        coefficients[idx+3] = 3
#for idx in range(len(coefficients)-3, len(coefficients)):
#    coefficients[idx] = 0.05
guesses = getGuessParams_model(templateRange_eD, nTemplates_eD, False)
for idx,entry in enumerate(guesses):
    if entry == 0:
        entry = 10.0
    coefficients[idx+3] = float(entry)
    

nll = lambda *args:-compoundLnlike(*args)

argsForNLL = [1]
for coeff in coefficients[3:]:
    argsForNLL.append(coeff)
#bounds = [(0.9,1.1)]
bounds = scaleLims
for i in range(nTemplates_eD):
#    bounds.append((-10, 11))
    bounds.append((float(0.), float(1.0e5)))
print('coefficient list length {}, type {}'.format(len(coefficients), type(coefficients)))
print(coefficients)
print('constraint list length {}, type {}'.format(len(bounds), type(bounds)))
print(bounds)
doML = True
if doML:
    optimizeRes = optimize.minimize(nll, coefficients, 
                                    args=(observedTOF,standoffs, tofRunBins, tof_range,shapeTemplates ),
                                    bounds=bounds, method='SLSQP', options={'disp': True, 'maxiter': 10000})
    optimCoeffs = optimizeRes["x"]
    print('Optimized coefficients that will be used:')
    print(optimCoeffs)
#for idx,optiCo in enumerate(optimCoeffs[1:]):
#    coefficients[idx+3] = optiCo
    coefficients = optimCoeffs

modelTest = []
for idx,templateSet in enumerate(shapeTemplates):
    if idx == 0:
        coeffs = [1]
    else:
        coeffs = [coefficients[idx-1]]
    for co in coefficients[3:]:
        coeffs.append(co)
    modelTest.append( buildModelTOF(coeffs, templateSet))

fig, (ax0, ax1, ax2, ax3) = plot.subplots(4, 2, sharex=False)
for idx,ax in enumerate([ax0, ax1, ax2, ax3]):
    ax[0].plot(observedTOFbinEdges[idx], observedTOF[idx], color='green')
    ax[0].set_ylabel('Experimental counts')
    axb = ax[0].twinx()
    axb.plot(observedTOFbinEdges[idx], modelTest[idx], color='red')
    axb.set_ylabel('Model counts')
    ax[1].scatter(observedTOFbinEdges[idx], (observedTOF[idx] - modelTest[idx]))
    ax[1].set_ylabel('Residual')


plot.figure()
plot.plot(templateGenVals_eD, coefficients[3:])
plot.ylim(0., max(coefficients[3:]) * 1.1)
plot.draw()

testLike = compoundLnlike(coefficients, observedTOF, standoffs, tofRunBins,
                          tof_range, shapeTemplates )

print(testLike)
#
#plot.show()
#quit()
#
nDim = 3 + nTemplates_eD
nWalkers = 500

p0 = [coefficients + 5e-4 * np.random.randn(nDim) for i in range(nWalkers)]
for walker in p0:
    for idx, par in enumerate(walker):
        if par <= 0:
            walker[idx] = 1

sampler = emcee.EnsembleSampler( nWalkers, nDim, lnprob, 
                                kwargs={'observables': observedTOF,
                                        'standoffs': standoffs,
                                        'tofbinnings': tofRunBins,
                                        'tofranges': tof_range,
                                        'templates': shapeTemplates},
                                        threads=4)
fout = open('burninchain.dat','w')


burninSteps = 2000
for i,samplerOut in enumerate(sampler.sample(p0, iterations=burninSteps)):
    burninPos, burninProb, burninRstate = samplerOut
    if i%50 == 0:
        print('burn-in step {} of {}'.format(i, burninSteps))
    if i%10 == 0: # only save every 10th step
        fout = open('burninchain.dat','a')
        for k in range(burninPos.shape[0]):
            fout.write('{} {} {}\n'.format(k, burninPos[k], burninProb[k]))
        fout.close()

# get the values to each coefficient...
coeffVals = []
for coeffNum in range(nTemplates_eD+3):
    coeffVals.append(sampler.chain[:,-500:,coeffNum].flatten())

coeffCentralVals = []
coeffStdDevs = []
coeffQuartiles = []
for cv in coeffVals:
    coeffCentralVals.append(np.mean(cv))
    coeffStdDevs.append(np.std(cv))
    coeffQuartiles.append(np.percentile(cv, [16,50,84], axis=0))
    
fig, (sfAx0, sfAx1, sfAx2) = plot.subplots(3)
sfAx0.hist(coeffVals[0])
sfAx0.vlines(coeffQuartiles[0][0], sfAx0.get_ylim()[0], sfAx0.get_ylim()[1], linestyles='dashed', colors='r')
sfAx0.vlines(coeffQuartiles[0][1:], sfAx0.get_ylim()[0], sfAx0.get_ylim()[1], linestyles='dotted', colors='r')
sfAx0.set_ylabel('Scale factor, run 1')
sfAx1.hist(coeffVals[1])
sfAx1.set_ylabel('Scale factor, run 2')
sfAx2.hist(coeffVals[2])
sfAx2.set_ylabel('Scale factor, run 3')
plot.draw()

#for scaleFactor in coeffCentralVals[:3]:    
#    plot.figure()
#    plot.hist( cv, bins=20 )
#    plot.draw()

   

plot.figure()
plot.errorbar(templateGenVals_eD, coeffCentralVals[3:], 
              yerr=coeffStdDevs[3:], fmt='k.')
plot.ylabel('Energy coefficient')
plot.title('Plot of energy component coefficients')
plot.xlabel('Energy (keV)')
plot.draw()

runX = np.linspace(1,3,3)
plot.figure()
plot.errorbar(runX, coeffCentralVals[:3], 
              yerr=coeffCentralVals[:3], fmt='k.')
plot.xlim(0,4)
plot.ylabel('Run weight coefficients')
plot.xlabel('Run')
plot.title('Plot of run weight coefficients')
plot.draw()

plot.show()
