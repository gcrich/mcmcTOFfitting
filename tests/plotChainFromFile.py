#!/home/gcr/anaconda3/bin/python
#
# plotChainFromFile.py
#
# pulls an MCMC chain from file and plots it
# convenient helper utility

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
from utilities.utilities import readChainFromFile
argParser = argparse.ArgumentParser()
argParser.add_argument('-file')
parsedArgs = argParser.parse_args()
chainFilename = parsedArgs.file

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
chain, probs, nParams, nWalkers, nSteps = readChainFromFile(chainFilename)
steps = np.linspace(1, nSteps, nSteps)
plt.figure()
for paramNum in range(nParams):
    plt.subplot((nParams + 1)*100 + 10 + paramNum+1)
    plt.plot( steps, chain[:,:,paramNum], color='k', alpha=0.2)
    plt.ylabel('Param {}'.format(paramNum))
plt.subplot((nParams+1)*100+10+nParams+1)
plt.plot( steps, probs[:,:], color='k', alpha=0.2)
plt.ylabel('ln probability')
plt.xlabel('Step')
plt.draw()

plt.figure()
plt.subplot(311)
plt.plot(steps, chain[:,:,0], color='k', alpha=0.2)
plt.ylabel('$E_0$ (keV)')
plt.subplot(312)
plt.plot(steps, chain[:,:,1], color='k', alpha=0.2)
plt.ylabel('$\sigma_0$')
plt.subplot(313)
plt.plot(steps, chain[:,:,2],color='k', alpha=0.2)
plt.ylabel('skew')
plt.xlabel('Step')
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

plt.figure()
plt.scatter(chain[:,:,0], probs[:,:], alpha=0.05)
plt.draw()

plt.show()
