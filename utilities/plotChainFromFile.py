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

argParser = argparse.ArgumentParser()
argParser.add_argument('-file')
parsedArgs = argParser.parse_args()
chainFilename = parsedArgs.file

chainList = []
probsList = []
indexList = []
with open(chainFilename,'r') as f:    
    for line in f.readlines():
        trimmedLine = line[line.find('[')+1:line.find(']')].split()
        vals = []
        for valStr in trimmedLine:
            vals.append(float(valStr))
        chainList.append(vals)
        probsList.append(float(line[line.find(']')+1:].rstrip('\n')))
        indexList.append(float(line[:line.find('[')]))
        
strideChains = []
probChains = []
for idx,entry in enumerate(chainList):
    if indexList[idx] == 0:
        strideChainList = []
        probChainList = []
    probChainList.append(probsList[idx])
    strideChainList.append(entry)
    if indexList[idx] == max(indexList):
        strideChains.append(strideChainList)
        probChains.append(probChainList)
        
chain = np.array(strideChains)
probs = np.array(probChains)

nSteps = int(len(indexList)/(max(indexList)+1))
steps = np.linspace(1, nSteps, nSteps)
plt.figure()
plt.subplot(311)
plt.plot(steps, chain[:,:,0], color='k', alpha=0.2)
plt.ylabel('$E_0$ (keV)')
plt.subplot(312)
plt.plot(steps, chain[:,:,1], color='k', alpha=0.2)
plt.ylabel('$\sigma_0$')
plt.subplot(313)
plt.plot(steps, probs[:,:],color='k', alpha=0.2)
plt.ylabel('ln probability')
plt.xlabel('Step')
plt.draw()
        