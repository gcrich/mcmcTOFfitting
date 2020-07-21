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
from utilities import ppcTools_oneBD
from initialization import initialize_oneBD
from scipy.stats import (skewnorm, lognorm)
from matplotlib import rcParams
import argparse


# list flattener from http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
flattenList = lambda l: [item for sublist in l for item in sublist]

defaultInputDataFilename = '/var/phy/project/phil/shared/quenchingFactors/csi_oneBD/tof/oneBD_CFD20_mcmcInputData.dat'

defaultChainDataFilename = '/home/gcr/Documents/quenchingFactors/csi_oneBD/tof/mcmcFits/cfdTiming_bestFit_15-07-2020/oneBD_mainchain.dat'

argParser = argparse.ArgumentParser()
argParser.add_argument('-file', default=defaultChainDataFilename, type=str)
argParser.add_argument('-inputDataFilename', default=defaultInputDataFilename, type=str)
argParser.add_argument('-shiftTOF', type=int, default=0)
argParser.add_argument('-lnprobcut', type=float, default=0.)
parsedArgs = argParser.parse_args()
chainFilename = parsedArgs.file
inputDataFilename = parsedArgs.inputDataFilename
tofShift = parsedArgs.shiftTOF
lnprobcut = parsedArgs.lnprobcut

# DEFINE BINNING AND SHIT
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

# tof_nBins = tofWindowSettings.nBins[standoffName[runNumber]]
# tof_minRange = tofWindowSettings.minRange[standoffName[runNumber]]
# tof_maxRange = tofWindowSettings.maxRange[standoffName[runNumber]]
# tof_range = (tof_minRange,tof_maxRange)

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


################################################
# binning set up

eD_bins, eD_range, eD_binSize, eD_binCenters = initialize_oneBD.setupDeuteronBinning()
x_bins, x_range, x_binSize, x_binCenters = initialize_oneBD.setupXbinning()

eD_minRange, eD_maxRange = eD_range
x_minRange, x_maxRange = x_range

eN_binCenters = getDDneutronEnergy( eD_binCenters )

################################################





ppcTool = ppcTools_oneBD.ppcTools_oneBD(chainFilename)

#chain, probs, nParams, nWalkers, nSteps = readChainFromFile(chainFilename)

print('have eD_bins value {}\n'.format(eD_bins))

chain = ppcTool.chain
probs = ppcTool.probs
nParams = ppcTool.nParams
nWalkers = ppcTool.nWalkers
nSteps = ppcTool.nSteps

steps = np.linspace(1, nSteps, nSteps)

returnVal = ppcTool.generatePPC(50, lnprobcut)
ppc = returnVal[0]

collectedPPCs = [ppc0 for ppc0 in ppc[0]]
for ppcDataSet in ppc[1:]:
    for idx, standoffData in enumerate(ppcDataSet):
        collectedPPCs[idx] = np.vstack((standoffData, collectedPPCs[idx]))

        
meansAllData = [np.mean(collectedPPC[:,:],axis=0) for collectedPPC in collectedPPCs]
statsAllData = [np.percentile(collectedPPC[:,:], [16,50,84], axis=0) for collectedPPC in collectedPPCs]

neutronSpectrumCollection = np.zeros(eD_bins)
neutronSpecPPCdata = returnVal[1]

neutronSpectrumCollectionNormalized = flattenList([[np.sum(subsubspec, axis=0)/np.sum(subsubspec) for subsubspec in subspec] for subspec in neutronSpecPPCdata])
neutronStats = np.percentile(neutronSpectrumCollectionNormalized, [16,50,84], axis=0)



dSpecPPCdata = returnVal[2]
dSpectrumCollection = flattenList([[np.sum(subsubspec, axis=0)/np.sum(subsubspec) for subsubspec in subspec] for subspec in dSpecPPCdata])
dSpectrum = np.sum(dSpectrumCollection, axis=0)
#print('shape of d spectrum samples is {}'.format(dSpectrumCollection[1:,:].shape))
dStats = np.percentile(dSpectrumCollection, [16,50,84], axis=0)

# get the data from file
tofData = readMultiStandoffTOFdata(inputDataFilename, 3)


if tofShift > 0:
    newTimeBins = tofData[:,0][:-tofShift]
    tofData = tofData[tofShift:,]
    tofData[:,0] = newTimeBins
if tofShift < 0:
    tofShift = -1*tofShift
    newTimeBins = tofData[:,0][tofShift:]
    tofData = tofData[:-tofShift,]
    tofData[:,0] = newTimeBins

binEdges = tofData[:,0]



observedTOF = []
observedTOFbinEdges=[]
for i in range(nRuns):
    observedTOF.append(tofData[:,i+1][(binEdges >= tof_minRange[i]) & (binEdges < tof_maxRange[i])])
    observedTOFbinEdges.append(tofData[:,0][(binEdges>=tof_minRange[i])&(binEdges<tof_maxRange[i])])


runColors = ["#ff678f", "#004edc", "#e16400"] 
runNames=['Close', 'Middle', 'Far']         
tofXvals = [np.linspace(minT, maxT, bins) for minT, maxT, bins in zip(tof_minRange, tof_maxRange, tofRunBins)]
fig, axes = plt.subplots(nRuns, figsize =(8.5,10.51))
plotIndexOrder = [0,1,2]
#ylims=[30e3,12e3,22e3,35e3]
#xlims=[(130,175),(130,175),(175,225),(190,260)]
for idx in range(nRuns):
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
    #ax.set_ylim(0., ylims[idx])
    #ax.set_xlim(xlims[idx])
    #ax.set_ylim(0, ylims[idx])
    #ax.set_xbound(ppcTool.tof_minRange[plotIndexOrder[idx]], 
     #           ppcTool.tof_minRange[plotIndexOrder[idx]])
#plt.tight_layout()
plt.xlabel('Time of flight (ns)')
plt.draw()
plt.savefig('PPC_on_data.png',dpi=300)

sdef, enBins, enHist = ppcTool.makeSDEF_sia_cumulative()

def rebin(origHist, origBins, rebinFactor = 2):
    evenBinContents = origHist[0::2]
    oddBinContents = origHist[1::2]
    rebinnedContents = evenBinContents + oddBinContents

    evenBinCenters = origBins[0::2]
    oddBinCenters = origBins[1::2]
    newBinCenters = (evenBinCenters + oddBinCenters)/2
    return rebinnedContents, newBinCenters

rebinnedHist, rebinnedCenters = rebin(enHist, enBins)
rebinnedHist, rebinnedCenters = rebin(rebinnedHist, rebinnedCenters)
# truncate these so a bunch of zero bins arent included
firstZero = np.where(rebinnedHist ==0)[0][0]
rebinnedHist_truncated = rebinnedHist[:firstZero]
rebinnedCenters_truncated = rebinnedCenters[:firstZero]

# write csv file with rebinned neutron energy distribution
# format is <bin center>, <bin content>
# bin contents are normalized so that integral is 1
histIntegral = np.sum(rebinnedHist)
with open('neutronEnergyDist.csv', 'w') as distCsvFile:
    for binCenter, binContent in zip(rebinnedCenters, rebinnedHist):
        distCsvFile.write('{:.3f}, {:.3e}\n'.format(binCenter/1000., 
            binContent / histIntegral))

siStrings = ['si{} a'.format(100)]
spStrings = ['sp{}'.format(100)]
for eN, counts in zip(rebinnedCenters_truncated, rebinnedHist_truncated):
            siStrings.append(' {:.3f}'.format(eN/1000))
            spStrings.append(' {:.3e}'.format(counts/1e7))
siString = ''.join(siStrings)
spString = ''.join(spStrings)
with open('sdefout.txt','w') as file:
    file.write(siString +'\n')
    file.write(spString)

fig, ax = plt.subplots(figsize=(8.5,5.25))
#ax.scatter(enBins, enHist/1.e11)
ax.scatter(rebinnedCenters, rebinnedHist/np.sum(rebinnedHist))
ax.set_xlim(3.e3, 5.5e3)
ax.set_ylabel('Intensity (arb units)')
ax.set_xlabel('Neutron energy (keV)')
plt.draw()
plt.savefig('neutronEnergyDist.png', dpi=300)