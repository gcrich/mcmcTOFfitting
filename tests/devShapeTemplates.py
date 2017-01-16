#
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
        eZeros = np.repeat(params, nEvPerLoop)
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
    
nTemplates_eD = 25
templateRange_min_eD, templateRange_max_eD = 600, 1200
templateRange_eD = (templateRange_min_eD, templateRange_max_eD)
templateGenVals_eD = np.linspace(templateRange_min_eD, templateRange_max_eD, 
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

def lnlike(params, observables, templates):
    modelTOF = buildModelTOF(params, templates)
    if not np.isfinite(np.sum(modelTOF)):
        return -np.inf
    obsErr = np.sqrt(observables)
    inverse_sigma2 = 1.0/(obsErr**2 + modelTOF**2*np.exp(2*np.log(2)))
    for i in range(len(observables)):
        pointLike = -0.5 * (observables[i] - modelTOF[i])**2 * inverse_sigma2[i] -np.log(inverse_sigma2[i])
        print('observable {} model {} 1/sigma2 {} likelihood {}'.format(observables[i], modelTOF[i], inverse_sigma2[i], pointLike))
    likelihood = -0.5*(np.sum((observables - modelTOF)**2 * inverse_sigma2 - np.log(inverse_sigma2)))
    return likelihood
    
def compoundLnlike(theta, observables, standoffs, tofbinnings, 
                   tofranges, templates):
    args = [1]
    for coeff in theta[3:]:
        args.append(coeff)
    loglike = lnlike(args, observables[0], templates[0])
    for idx, scale in enumerate(theta[:3]):
        args = [scale]
        for coeff in theta[3:]:
            args.append(coeff)
        loglike = loglike + lnlike( args, observables[idx+1], templates[idx+1])
    return loglike
    
    
scaleLims = [(1.0, 1.5),(0.25, 0.75),(1.3, 1.9)]
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
        if cv < 0.0 or cv > 3.0:
#            print('coeff tripped inf')
#            print(theta)
            return -inf
    return 0.0
    
def lnprob(theta, observables, standoffs, tofbinnings, tofranges, templates):
    prior = lnprior(theta)
    loglike = compoundLnlike(theta, observables, standoffs, 
                                 tofbinnings, tofranges, templates)
    return prior + loglike

    
    
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
            print('generating template for standoff {} of energy {}'.format(runIndex, energyIdx))
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
    coefficients[i-1] = tofCounts[i]/tofCounts[0]

#initialIndicesForGuess= []
for idx,entry in enumerate(templateGenVals_eD):
    if entry >= 650 and entry <= 950:
        coefficients[idx+3] = 0.025

nll = lambda *args:-lnlike(*args)

argsForNLL = [1]
for coeff in coefficients[3:]:
    argsForNLL.append(coeff)
bounds = [(0.9,1.1)]
for i in range(25):
    bounds.append((0, 5))
#optimizeRes = optimize.minimize(nll, argsForNLL, 
#                                args=(observedTOF[0],shapeTemplates[0]))
#                                #bounds=bounds, method='TNC')
#optimCoeffs = optimizeRes["x"]
#print('Optimized coefficients that will be used:')
#print(optimCoeffs[1:])
#for idx,optiCo in enumerate(optimCoeffs[1:]):
#    coefficients[idx+3] = optiCo / optimCoeffs[0]

modelTest = []
for idx,templateSet in enumerate(shapeTemplates):
    if idx == 0:
        coeffs = [1]
    else:
        coeffs = [coefficients[idx-1]]
    for co in coefficients[3:]:
        coeffs.append(co)
    modelTest.append( buildModelTOF(coeffs, templateSet))

fig, (ax0, ax1, ax2, ax3) = plot.subplots(4, sharex=False)
for idx,ax in enumerate([ax0, ax1, ax2, ax3]):
    ax.plot(observedTOFbinEdges[idx], observedTOF[idx], color='green')
    ax.set_ylabel('Experimental counts')
    axb = ax.twinx()
    axb.plot(observedTOFbinEdges[idx], modelTest[idx], color='red')
    axb.set_ylabel('Model counts')


plot.figure()
plot.plot(templateGenVals_eD, coefficients[3:])
plot.ylim(0., max(coefficients[3:]) * 1.1)
plot.draw()

testLike = compoundLnlike(coefficients, observedTOF, standoffs, tofRunBins,
                          tof_range, shapeTemplates )

print(testLike)
plot.show()
quit()

nDim = 3 + nTemplates_eD
nWalkers = 250

p0 = [coefficients + 5e-4 * np.random.randn(nDim) for i in range(nWalkers)]

sampler = emcee.EnsembleSampler( nWalkers, nDim, lnprob, 
                                kwargs={'observables': observedTOF,
                                        'standoffs': standoffs,
                                        'tofbinnings': tofRunBins,
                                        'tofranges': tof_range,
                                        'templates': shapeTemplates},
                                        threads=3)
fout = open('burninchain.dat','w')


burninSteps = 1500
for i,samplerOut in enumerate(sampler.sample(p0, iterations=burninSteps)):
    burninPos, burninProb, burninRstate = samplerOut
    if i%50 == 0:
        print('burn-in step {} of {}'.format(i, burninSteps))
    for k in range(burninPos.shape[0]):
        fout.write('{} {} {}\n'.format(k, burninPos[k], burninProb[k]))
fout.close()

# get the values to each coefficient...
coeffVals = []
for coeffNum in range(nTemplates_eD+3):
    coeffVals.append(sampler.chain[:,-500:,coeffNum].flatten())

coeffCentralVals = []
coeffStdDevs = []
for cv in coeffVals:
    coeffCentralVals.append(np.mean(cv))
    coeffStdDevs.append(np.std(cv))
    plot.figure()
    plot.hist( cv, bins=20 )
    plot.draw()

   

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
