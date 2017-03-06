from __future__ import print_function
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from constants.constants import (masses, distances, physics, tofWindows)
from utilities.utilities import (beamTimingShape, ddnXSinterpolator,
                                 getDDneutronEnergy, readChainFromFile,
                                 getTOF, readMultiStandoffTOFdata)
from utilities.ionStopping import ionStopping
from utilities.ppcTools import ppcTools
from scipy.stats import (skewnorm, lognorm)
import argparse


# list flattener from http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
flattenList = lambda l: [item for sublist in l for item in sublist]



argParser = argparse.ArgumentParser()
argParser.add_argument('-file')
argParser.add_argument('-tofDataFile')
argParser.add_argument('-nParamSamples', default=50, type=int)
argParser.add_argument('-nBinsE', default=100, type=int)
argParser.add_argument('-nBinsX', default=20, type=int)
argParser.add_argument('-nRuns', default=4, type=int) # number of runs data addresses
parsedArgs = argParser.parse_args()
chainFilename = parsedArgs.file
tofDataFilename = parsedArgs.tofDataFile
nParamSamples = parsedArgs.nParamSamples
nBinsE = parsedArgs.nBinsE
nBinsX = parsedArgs.nBinsX
numRuns = parsedArgs.nRuns

ppcTool = ppcTools(chainFilename, nSamplesFromTOF=5000,
                   nBins_eD = nBinsE, nBins_x = nBinsX, nRuns = numRuns)
#ppc, ppcNeutronSpectra = ppcTool.generatePPC( nChainEntries = nParamSamples )
returnVal = ppcTool.generatePPC( nChainEntries = nParamSamples )
ppc = returnVal[0]

collectedPPCs = [ppc0 for ppc0 in ppc[0]]
for ppcDataSet in ppc[1:]:
    for idx, standoffData in enumerate(ppcDataSet):
        collectedPPCs[idx] = np.vstack((standoffData, collectedPPCs[idx]))

        
meansAllData = [np.mean(collectedPPC[:,:],axis=0) for collectedPPC in collectedPPCs]
statsAllData = [np.percentile(collectedPPC[:,:], [16,50,84], axis=0) for collectedPPC in collectedPPCs]

neutronSpectrumCollection = np.zeros(ppcTool.eD_bins)
neutronSpecPPCdata = returnVal[1]
#for sampledParamSet in neutronSpecPPCdata:
#    samplesAlongLength = sampledParamSet[0]
#    summedAlongLength = np.sum(samplesAlongLength, axis=0)
#    neutronSpectrumCollection = np.vstack((neutronSpectrumCollection, summedAlongLength))
#neutronSpectrum = np.sum(neutronSpectrumCollection, axis=0)
#neutronSpectrumCollectionNormalizedList = []
#for spectrum in neutronSpectrumCollection[1:,:]:
#    normFactor = np.sum(spectrum)
#    neutronSpectrumCollectionNormalizedList.append(spectrum / normFactor)
#neutronSpectrumCollectionNormalized = np.array(neutronSpectrumCollectionNormalizedList)
neutronSpectrumCollectionNormalized = flattenList([[np.sum(subsubspec, axis=0)/np.sum(subsubspec) for subsubspec in subspec] for subspec in neutronSpecPPCdata])
neutronStats = np.percentile(neutronSpectrumCollectionNormalized, [16,50,84], axis=0)
             

#dSpectrumCollection = np.zeros(ppcTool.eD_bins)
dSpecPPCdata = returnVal[2]
##dSpecNormFactors = [[np.sum(dSpecUnder) for dSpecUnder in dSpec] for dSpec in dSpecPPCdata]
##flattenedDspec = flattenList(dSpecPPCdata)
##dSpecNorms = np.sum(np.sum(flattenedDspec, axis=1), axis=1)
#for sampledParamSet in dSpecPPCdata:
#    samplesAlongLength = sampledParamSet[0]
#    summedAlongLength = np.sum(samplesAlongLength, axis=0)
#    dSpectrumCollection = np.vstack((dSpectrumCollection, summedAlongLength))
dSpectrumCollection = flattenList([[np.sum(subsubspec, axis=0)/np.sum(subsubspec) for subsubspec in subspec] for subspec in dSpecPPCdata])
dSpectrum = np.sum(dSpectrumCollection, axis=0)
#print('shape of d spectrum samples is {}'.format(dSpectrumCollection[1:,:].shape))
dStats = np.percentile(dSpectrumCollection, [16,50,84], axis=0)


sdef_sia_cumulative = ppcTool.makeSDEF_sia_cumulative()
print('got SDEF...')
print('{}'.format(sdef_sia_cumulative))   
 
# get the data from file
tofData = readMultiStandoffTOFdata(filename=tofDataFilename, nRuns=numRuns)


binEdges = tofData[:,0]               

observedTOF = []
observedTOFbinEdges=[]
for i in range(numRuns):
    observedTOF.append(tofData[:,i+1][(binEdges >= ppcTool.tof_minRange[i]) & (binEdges < ppcTool.tof_maxRange[i])])
    observedTOFbinEdges.append(tofData[:,0][(binEdges>=ppcTool.tof_minRange[i])&(binEdges<ppcTool.tof_maxRange[i])])
                
# this is from colorbrewer2
#runColors=['#bf5b17','#386cb0','#beaed4','#7fc97f', '#fdc086']  
# next set is from iwanthue  - http://tools.medialab.sciences-po.fr/iwanthue/
# last color is good for overlaying lines
runColors = ["#ff678f", "#004edc", "#e16400", "#1fceff", "#781616", "#149c00"] 
runNames=['Middle','Close', 'Close (detuned)', 'Far', 'Production']         
tofXvals = [np.linspace(minT, maxT, bins) for minT, maxT, bins in zip(ppcTool.tof_minRange, ppcTool.tof_maxRange, ppcTool.tofRunBins)]
fig, axes = plt.subplots(numRuns)
plotIndexOrder = [1,2,0,3,4]
ylims=[30e3,40e3,15e3,40e3]
for idx in range(numRuns):
    index = plotIndexOrder[idx]
    ax=axes[idx]
    ax.scatter(tofXvals[plotIndexOrder[idx]], 
               observedTOF[plotIndexOrder[idx]], 
               color=runColors[plotIndexOrder[idx]], 
               label=runNames[plotIndexOrder[idx]])
    ax.plot(tofXvals[plotIndexOrder[idx]],
            statsAllData[plotIndexOrder[idx]][1,:], color='#f0027f')
    ax.fill_between(tofXvals[plotIndexOrder[idx]],
            statsAllData[plotIndexOrder[idx]][0,:],
            statsAllData[plotIndexOrder[idx]][2,:],
            facecolor='#f0027f', alpha=0.3, linewidth=0)
    #ax.plot(tofXvals[plotIndexOrder[idx]],
     #       statsAllData[plotIndexOrder[idx]][2,:], color='red', alpha=0.4)
    ax.set_ylabel('Counts')
    ax.legend(loc='upper right')
    #ax.set_ylim(0, ylims[idx])
    #ax.set_xbound(ppcTool.tof_minRange[plotIndexOrder[idx]], 
     #           ppcTool.tof_minRange[plotIndexOrder[idx]])
#plt.tight_layout()
plt.xlabel('Time of flight (ns)')
plt.draw()
plt.savefig('PPC_on_data.png', dpi=400)




# make a plot of initial D energy distribution
dZeroSamples = ppcTool.sampleInitialEnergyDist(nSamples=nParamSamples)
dZeroSamplesNormed = [dZero / np.sum(dZero) for dZero in dZeroSamples]
#print('shape of d0 samples is {}'.format(dZeroSamples.shape))
dZeroStats= np.percentile(dZeroSamplesNormed, [16,50,84], axis=0)



dColors = ['#a6611a','#dfc27d']
nColors = ['#018571','#80cdc1']
eN_xVals = ppcTool.eN_binCenters
fig, axNeutrons = plt.subplots()

nBand= axNeutrons.fill_between(eN_xVals, neutronStats[0,:], neutronStats[2,:],
                        facecolor=nColors[0], linewidth=0, alpha=0.4,
                        label='Neutron energies')
axNeutrons.plot(eN_xVals, neutronStats[1,:], color=nColors[0] )
axNeutrons.tick_params('x', colors=nColors[0])
axNeutrons.set_xlabel('Neutron energy (keV)', color=nColors[0])
axNeutrons.set_ylabel('Intensity')

axDeuterons = axNeutrons.twiny()#.twinx()

dBand = axDeuterons.fill_between(ppcTool.eD_binCenters, dStats[0,:], dStats[2,:],
                         facecolor=dColors[0], linewidth=0, alpha=0.4,
                         label='Deuteron energies\nafter transport')
axDeuterons.plot(ppcTool.eD_binCenters, dStats[1,:], color=dColors[0])
axDeuterons.set_xlabel('Deuteron energy (keV)', color=dColors[0])
axDeuterons.tick_params('x', colors=dColors[0])
#axDZero = axDeuterons.twinx()
dZBand = axDeuterons.fill_between(ppcTool.eD_binCenters, dZeroStats[0,:], 
                         dZeroStats[2,:], facecolor=dColors[1], linewidth=0,
                         alpha=0.4, label='Initial deuteron energies')
axDeuterons.plot(ppcTool.eD_binCenters, dZeroStats[1,:], color=dColors[1])
#axDZero.set_ylabel('Initial deuteron intensity', color=dColors[1])
#axDZero.tick_params('y', colors=dColors[1])
#axNeutrons.plot(eN_xVals, neutronStats[2,:], color=nColors[1], alpha=0.4)

#axNeutrons.tick_params('y', colors=nColors[0])

#axDeuterons.plot(ppcTool.eD_binCenters, dStats[2,:], color=dColors[1], alpha=0.4)
#axDeuterons.set_ylabel('Counts (a.u.)')
features = [dZBand, dBand, nBand]
labels = [feat.get_label() for feat in features]
axNeutrons.legend(features, labels, loc=0)
plt.draw()
plt.savefig('PPC_neutronSpec.png', dpi=400)






#ppcTool.makeCornerPlot(plotFilename = 'corner_allParams.png')
#ppcTool.makeCornerPlot(paramIndexHigh = 4, plotFilename = 'corner_eParams.png')

plt.show()