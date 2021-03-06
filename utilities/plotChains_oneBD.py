#!/home/gcr/anaconda3/bin/python
#
# plotChains_oneBD.py
#
# pulls an MCMC chain from file and plots it
# convenient helper utility
#
# note that some hacked edits were made to prepare plots for GCR thesis...
# HACKED TOGETHER FROM "plotChainsFromFile.py"

from __future__ import print_function
import numpy as np
from numpy import inf
import scipy.optimize as optimize
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import emcee
import csv as csvlib
import argparse
from numbers import Number
#import readChainFromFile
#from utilities import readChainFromFile
#from utilities import ppcTools_oneBD
import ppcTools_oneBD




argParser = argparse.ArgumentParser()
argParser.add_argument('-file')
argParser.add_argument('-lnprobCut',type=float, default=0., help="Only show steps below this LNprob value")
parsedArgs = argParser.parse_args()
chainFilename = parsedArgs.file
lnprobCut = parsedArgs.lnprobCut


parameterLabels = [r'$E_{beam}$ (keV)', r'$\theta$ (keV)', r'$m$ (keV)', r'$\sigma$ (1 / keV)']

#chainList = []
#probsList = []
#indexList = []
#maxIndex = 0
#with open(chainFilename,'r') as f:
#    line = f.readline()
#    if len(line) == 0:
#        keepReading = False
#    keepReading = True    
#    while keepReading:
#        indexVal = float(line[:line.find('[')])
#        indexList.append(indexVal)
#        if indexVal > maxIndex:
#            maxIndex = indexVal
#        paramWrap = True
#        vals = []
#        paramWrap=False
#        endValsIndex = line.find(']')
#        if endValsIndex == -1:
#            # in this case, parameters keep going onto next line
#            endValsIndex = len(line)
#            paramWrap = True
#        trimmedLine = line[line.find('[')+1:endValsIndex].split()
#        for valStr in trimmedLine:
#            vals.append(float(valStr))
#        while paramWrap:  
#            line = f.readline()
#            endValsIndex = line.find(']')
#            if endValsIndex == -1:
#                # in this case, parameters keep going onto next line
#                endValsIndex = len(line)
#                paramWrap = True
#            else:
#                paramWrap = False
#            trimmedLine = line[:endValsIndex].split()
#            for valStr in trimmedLine:
#                vals.append(float(valStr))
#        chainList.append(vals)
#        probsList.append(float(line[line.find(']')+1:].rstrip('\n')))
#        line = f.readline()
#        if len(line) == 0:
#            keepReading = False
#     
#            
#strideChains = []
#probChains = []
#for idx,entry in enumerate(chainList):
#    if indexList[idx] == 0:
#        strideChainList = []
#        probChainList = []
#    probChainList.append(probsList[idx])
#    strideChainList.append(entry)
#    if indexList[idx] == maxIndex: #should be == max(indexList)
#        strideChains.append(strideChainList)
#        probChains.append(probChainList)
#        
#chain = np.array(strideChains)
#probs = np.array(probChains)
#
#nParams = len(chain[0,0])
#nSteps = int(len(indexList)/(max(indexList)+1))

thePPCobject = ppcTools_oneBD.ppcTools_oneBD(chainFilename)

#chain, probs, nParams, nWalkers, nSteps = readChainFromFile(chainFilename)


chain = thePPCobject.chain
probs = thePPCobject.probs
nParams = thePPCobject.nParams
nWalkers = thePPCobject.nWalkers
nSteps = thePPCobject.nSteps

steps = np.linspace(1, nSteps, nSteps)

if lnprobCut != 0.:
    probCutIndices = np.where(probs > lnprobCut)
    print('probcut {}\n'.format(probCutIndices))
    print('probcut len {}\n'.format(len(probCutIndices)))
    print('probcut len 0 {}\n'.format(len(probCutIndices[0])))
    print('probcut len 1 {}\n'.format(len(probCutIndices[1])))
    probCutSteps, probCutWalkers = probCutIndices
    # i think the format here is probcutIndices[0] -> sample
    # probcutIndices[1] -> walker

#plt.figure()
#for paramNum in range(nParams):
#    plt.subplot((nParams + 1)*100 + 10 + paramNum+1)
#    plt.plot( steps, chain[:,:,paramNum], color='k', alpha=0.2)
#    plt.ylabel('Param {}'.format(paramNum))
#plt.subplot((nParams+1)*100+10+nParams+1)
#plt.plot( steps, probs[:,:], color='k', alpha=0.2)
#plt.ylabel('ln probability')
#plt.xlabel('Step')
#plt.draw()

# TODO: there are some pretty hardcoded things here
#       should make this as general as possible, but for now i need to graduate 
#       - so things are hacked together to make plots necessary for thesis
fig, axes = plt.subplots(4, sharex=True, figsize =(8.5,10.51))
for ax, chainSamples, label in zip(axes, 
                            [chain[:,:,0],chain[:,:,1], chain[:,:,2], chain[:,:,3]],
                            parameterLabels):#[r'$E_{beam}$ (keV)',r'$f_1$',r'$f_2$',r'$f_3$']):
    ax.plot(steps, chainSamples, alpha=0.2, color='k')
    ax.set_ylabel(label)
# if lnprobCut != 0:
#     for ax, chainSamples, label in zip(axes,
#     [chain[probCutIndices[0], probCutIndices[1],0], chain[probCutIndices[0],probCutIndices[1],1], chain[probCutIndices[0],probCutIndices[1],2], chain[probCutIndices[0], probCutIndices[1], 3], [r'$E_{beam}$ (keV)', r'$\theta$ (keV)', r'$m$ (keV)', r'$\sigma$ (1 / keV)']):
#         ax.plot(steps, chainSamples, alpha=0.3, color='r')

#axes[0].plot(steps, chain[:,:,0], color='k', alpha=0.2)
#plt.ylabel('$E_{beam}$ (keV)')
#plt.subplot(412)
#plt.plot(steps, chain[:,:,1], color='k', alpha=0.2)
#plt.ylabel('$f_1$')
#plt.subplot(413)
#plt.plot(steps, chain[:,:,2],color='k', alpha=0.2)
#plt.ylabel('$f_2$')
#plt.subplot(414)
#plt.plot(steps, chain[:,:,3],color='k', alpha=0.2)
#plt.ylabel('$f_3$')
axes[3].set_xlabel('Step')
axes[3].set_xlim(0,nSteps)
plt.draw()
plt.savefig('oneBD_energyParameter_chain.pdf')

# for idx,chainSamples in enumerate(chain[:,:,4:]):
#     fig, ax = plt.subplots(figsize=(7,4))
#     print('shape steps {} shape chainSamples {} shape chainSamples[:,:] {}\n'.format(steps.shape, chainSamples.shape, chainSamples[:,:].shape))
#     ax.plot(steps, chainSamples[:,:], alpha=0.2, color='k')
#     ax.set_xlim(0,nSteps)
#     ax.set_xlabel('Step')
#     ax.set_ylabel('Par {}'.format(idx))
#     plt.draw()
chainSample = chain[:,:,4]
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(steps, chainSample, alpha=0.2, color='k')
plt.draw()

plt.figure()
plt.plot(steps, probs[:,:], color='k', alpha=0.2)
plt.ylabel('ln prob')
plt.xlabel('step')
plt.draw()
        
paramHists = []
paramHistBins = []
for paramNum in range(nParams):
    hist, bins = np.histogram(chain[:,:,paramNum], bins=100)
    paramHists.append( hist )
    paramHistBins.append(bins)
plt.figure()
plt.scatter(bins[:-1],hist)
plt.draw()

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey=True)
for ax, chainsub, parlabel in zip([ax1,ax2,ax3,ax4], [chain[:,:,0], chain[:,:,1], chain[:,:,2], chain[:,:,3]], parameterLabels):
        ax.scatter(chainsub, probs[:,:], alpha=0.05, c='k', marker=',')
        ax.set_xlabel(parlabel)
plt.draw()
plt.savefig('lnprob_params.png',dpi=300)

plt.show()

thePPCobject.makeCornerPlot(plotFilename='oneBD_corner.png')