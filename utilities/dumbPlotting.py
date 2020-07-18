#
# dumbPlotting.py
#
# meant to help me visualize distributions

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
from math import isnan
import gc
import matplotlib.pyplot as plt

#nBins = 600
binWidth = 10
range_min, range_max = (0, 1500)

bins = np.arange(range_min, range_max, binWidth)
nBins = len(bins)
binCenters = np.linspace( range_min + binWidth/2, range_max - binWidth/2, nBins, True)

beamE = 2450.
par1, par2, par3 = (1400, 50, 0.4)

nEvs = 20000
eZeros = np.repeat(beamE, nEvs)
eZeros -= lognorm.rvs(s=par3, loc=par1, scale=par2, size=nEvs)

par1, par2, par3 = (1400, 100, 1.0)
eZeros_1 = np.repeat(beamE, nEvs)
eZeros_1 -= lognorm.rvs(s=par3, loc=par1, scale=par2, size=nEvs)

par1, par2, par3 = (1400, 50, 1.)
eZeros_2 = np.repeat(beamE, nEvs)
eZeros_2 -= lognorm.rvs(s=par3, loc=par1, scale=par2, size=nEvs)

hist, edges = np.histogram(eZeros, len(bins), (range_min, range_max))
hist2, edges = np.histogram(eZeros_1, len(bins), (range_min, range_max))
hist_2, edges = np.histogram(eZeros_2, len(bins), (range_min, range_max))

fig, ax = plt.subplots(figsize=(8.5,5.25))
ax.scatter( binCenters, hist, color='k' )
ax.scatter( binCenters, hist2, color='red')
ax.scatter( binCenters, hist_2, color='blue')

plt.draw()