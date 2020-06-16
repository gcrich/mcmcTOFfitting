#
# testExpoConvolution.py
#
# just meant to demo spreading (tof) spectrum with exponential
# intended to mimic spread of transit across 0deg

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
from math import isnan
import gc
import matplotlib.pyplot as plt

from scipy import signal

nBins_tof = 100
tofRange_min, tofRange_max = 0, 400

tofBinCenters = np.linspace(2, 398, nBins_tof)

#fakePureDist = np.random.normal(200, 10, 5000)
fakePureDist= np.repeat(202, 2000)

tofHist, binEdges = np.histogram(fakePureDist, nBins_tof, (tofRange_min, tofRange_max))

fig,ax = plt.subplots(figsize=(8.5, 5.25))
ax.scatter(tofBinCenters, tofHist)
plt.draw()

expoBinCenters = np.linspace(0,24,7, True)
spreadVal = np.exp(-expoBinCenters/4)/np.sum(np.exp(-expoBinCenters/4))
convoKernel = spreadVal#[::-1]

fig, ax = plt.subplots(figsize=(8.5,5.25))
ax.scatter(expoBinCenters,spreadVal)
plt.draw()

sigConvolved = signal.convolve(tofHist, spreadVal, mode='same')

convolved = np.convolve(tofHist, convoKernel, 'same')
fullconvolved = np.convolve(tofHist, convoKernel, 'full')
fig,ax = plt.subplots(figsize=(8.5,5.25))
ax.scatter(tofBinCenters, tofHist, color='k')
ax.scatter(tofBinCenters, convolved, color='red', alpha=0.5)
#ax.scatter(tofBinCenters, sigConvolved, color='blue')
ax.scatter(tofBinCenters, fullconvolved[:-len(expoBinCenters)+1], color='red')
ax.set_xlim(150,250)
plt.draw()

gaus_binCenters = np.linspace(-20,20,11,True)
gaus_sigma=4
tempSpread = np.exp(-(gaus_binCenters / gaus_sigma)**2/2)
tempSpread = tempSpread / np.sum(tempSpread)
gausConv = np.convolve(tofHist, tempSpread, 'same')

fig, ax = plt.subplots(figsize=(8.5,5.25))
ax.scatter(tofBinCenters, tofHist, color='k')
ax.scatter(tofBinCenters, gausConv, color='red')
ax.set_xlim(150,250)
plt.draw()