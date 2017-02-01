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

argParser = argparse.ArgumentParser()
argParser.add_argument('-file')
argParser.add_argument('-tofDataFile')
argParser.add_argument('-nParamSamples', default=50, type=int)
argParser.add_argument('-nBinsE', default=100, type=int)
argParser.add_argument('-nBinsX', default=20, type=int)
parsedArgs = argParser.parse_args()
chainFilename = parsedArgs.file
tofDataFilename = parsedArgs.tofDataFile
nParamSamples = parsedArgs.nParamSamples
nBinsE = parsedArgs.nBinsE
nBinsX = parsedArgs.nBinsX

ppcTool = ppcTools(chainFilename, nSamplesFromTOF=5000,
                   nBins_eD = nBinsE, nBins_x = nBinsX)
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
for sampledParamSet in neutronSpecPPCdata:
    samplesAlongLength = sampledParamSet[0]
    summedAlongLength = np.sum(samplesAlongLength, axis=0)
    neutronSpectrumCollection = np.vstack((neutronSpectrumCollection, summedAlongLength))
neutronSpectrum = np.sum(neutronSpectrumCollection, axis=0)
neutronStats = np.percentile(neutronSpectrumCollection[1:,:], [16,50,84], axis=0)
             

sdef_sia_cumulative = ppcTool.makeSDEF_sia_cumulative()
print('got SDEF...')
print('{}'.format(sdef_sia_cumulative))   
 
# get the data from file
tofData = readMultiStandoffTOFdata(tofDataFilename)


binEdges = tofData[:,0]               

observedTOF = []
observedTOFbinEdges=[]
for i in range(4):
    observedTOF.append(tofData[:,i+1][(binEdges >= ppcTool.tof_minRange[i]) & (binEdges < ppcTool.tof_maxRange[i])])
    observedTOFbinEdges.append(tofData[:,0][(binEdges>=ppcTool.tof_minRange[i])&(binEdges<ppcTool.tof_maxRange[i])])
                
runColors=['#1b9e77','#d95f02','#7570b3','#e7298a']            
tofXvals = [np.linspace(minT, maxT, bins) for minT, maxT, bins in zip(ppcTool.tof_minRange, ppcTool.tof_maxRange, ppcTool.tofRunBins)]
fig, axes = plt.subplots(4)
for idx, ax in enumerate(axes):
    ax.scatter(tofXvals[idx], observedTOF[idx], color=runColors[idx])
    ax.plot(tofXvals[idx], statsAllData[idx][1,:], color='blue')
    ax.plot(tofXvals[idx], statsAllData[idx][0,:], color='red', alpha=0.4)
    ax.plot(tofXvals[idx], statsAllData[idx][2,:], color='red', alpha=0.4)
plt.draw()
plt.savefig('PPC_on_data.png', dpi=400)




eN_xVals = ppcTool.eN_binCenters
plt.figure()
plt.plot(eN_xVals, neutronStats[1,:], color='blue' )
plt.plot(eN_xVals, neutronStats[0,:], color='red', alpha=0.4)
plt.plot(eN_xVals, neutronStats[2,:], color='red', alpha=0.4)
plt.draw()
plt.savefig('PPC_neutronSpec.png', dpi=400)

plt.show()