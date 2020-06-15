#
# want to see if we can approximate bethe-bloch stopping with a polynomial or something
# maybe this would allow us to speed up running and not need.. a supercomputer
#
# of course really we should actually profile the sampling
# not 100% certain that it is the DE solving that takes time, but it is a good candidate
#

from __future__ import print_function
import numpy as np
from numpy import inf
import scipy.optimize as optimize
from scipy.integrate import ode
from scipy.stats import (poisson, norm, lognorm)
import scipy.special as special
from scipy.interpolate import RectBivariateSpline
#from scipy.stats import skewnorm
import sys
import emcee
import csv as csvlib
import argparse
from numbers import Number
from constants.constants import (masses, distances, physics, tofWindows)
from utilities.utilities import (beamTimingShape, ddnXSinterpolator, 
                                 getDDneutronEnergy)
from utilities.utilities import zeroDegreeTimingSpread
from utilities.utilities import readMultiStandoffTOFdata
from utilities.ionStopping import ionStopping
from math import isnan
import gc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# range of eD expanded for seemingly higher energy oneBD neutron beam
eD_bins = 120
# if quickAndDirty == True:
#     eD_bins = 20
eD_minRange = 400.
eD_maxRange = 1200.0
eD_range = (eD_minRange, eD_maxRange)
eD_binSize = (eD_maxRange - eD_minRange)/eD_bins
eD_binCenters = np.linspace(eD_minRange + eD_binSize/2,
                            eD_maxRange - eD_binSize/2,
                            eD_bins)


x_bins = 10
# if quickAndDirty == True:
#     x_bins = 5
x_minRange = 0.0
x_maxRange = distances.tunlSSA_CsI_oneBD.cellLength
x_range = (x_minRange,x_maxRange)
x_binSize = (x_maxRange - x_minRange)/x_bins
x_binCenters = np.linspace(x_minRange + x_binSize/2,
                           x_maxRange - x_binSize/2,
                           x_bins)


# parameters for making the fake data...
nEvPerLoop = 1000
data_x = np.repeat(x_binCenters,nEvPerLoop)



ddnXSinstance = ddnXSinterpolator()


# PARAMETER BOUNDARIES
min_beamE, max_beamE = 1500.0, 2000.0 # see lab book pg54, date 1/24/16 - 2070 field of 139.091 mT gives expected Ed = 1.8784 MeV
min_eLoss, max_eLoss = 600.0,1000.0
min_scale, max_scale = 40.0, 300.0
min_s, max_s = 0.1, 1.2

beamE_guess = 1860.0 # initial deuteron energy, in keV
eLoss_guess = 850.0 # width of initial deuteron energy spread
scale_guess = 180.0
s_guess = 0.6


# stopping power model and parameters
stoppingMedia_Z = 1
stoppingMedia_A = 2
stoppingMedia_rho = 8.565e-5 # from red notebook, p 157
incidentIon_charge = 1
# IMPORTANT
# CHECK TO SEE IF THIS FACTOR OF 1e-3 IS NEEDED
stoppingMedia_meanExcitation = 19.2 * 1e-3 # FACTOR OF 1e-3 NEEDED?
# IMPORTANT
stoppingModelParams = [stoppingMedia_Z, stoppingMedia_A, stoppingMedia_rho,
                       incidentIon_charge, stoppingMedia_meanExcitation]
stoppingModel = ionStopping.simpleBethe( stoppingModelParams )

# hack shit
# just because i don't want to change code i copied and pasted below
params = [beamE_guess, eLoss_guess, scale_guess, s_guess, 1e4, 10]
beamE, eLoss, scale, s, scaleFactor, bgLevel = params
ddnXSfxn = ddnXSinstance
dedxfxn = stoppingModel.dEdx
    
eN_binCenters = getDDneutronEnergy( eD_binCenters )



dedxForODE = lambda x, y: dedxfxn(energy=y,x=x)

nSamples = 10000
nEvPerLoop = 1000


eZero = beamE - lognorm.rvs(s=s, loc=eLoss, scale=scale)
solver = ode(dedxForODE).set_integrator('dopri5').set_initial_value(eZero)
solved = np.array([solver.integrate(x) for x in x_binCenters])

eDstep = 50
edgrid = np.arange(eD_minRange, eD_maxRange, eDstep)

x,y = np.meshgrid(x_binCenters, edgrid)
z = np.zeros(len(x_binCenters))
for eZero in edgrid:
    solver = ode(dedxForODE).set_integrator('dopri5').set_initial_value(eZero)
    thisSolution = np.array([solver.integrate(x) for x in x_binCenters])
    z = np.vstack((z,thisSolution.flatten()))
z = z[1:]

interp_spline = RectBivariateSpline(edgrid, x_binCenters, z)

eDstep_fine = 10
edgrid_fine = np.arange(eD_minRange, eD_maxRange, eDstep_fine)
x_fine, y_fine = np.meshgrid(x_binCenters, edgrid_fine)
#z_fine = interp_spline(y_fine, x_fine)
z_fine = np.zeros(len(x_binCenters))
for eZero in edgrid_fine:
    thisSolution = interp_spline(eZero, x_binCenters).flatten()
    z_fine = np.vstack((z_fine, thisSolution))
z_fine = z_fine[1:]

fig, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={'projection':'3d'})
ax[0].plot_wireframe(x, y, z)
ax[1].plot_wireframe(x_fine, y_fine, z_fine)
fig.tight_layout()
plt.draw()
plt.show()

dataHist = np.zeros((x_bins, eD_bins))
unweightedDataHist = np.zeros((x_bins, eD_bins))

dataHist_spline = np.zeros((x_bins, eD_bins))
unweightedDataHist_spline = np.zeros((x_bins, eD_bins))

nLoops = int(np.ceil(nSamples / nEvPerLoop))
nLoops = 1

for loopNum in range(0, nLoops):
    #eZeros = np.random.normal( params[0], params[0]*params[1], nEvPerLoop )
    #eZeros = skewnorm.rvs(a=skew0, loc=e0, scale=e0*sigma0, size=nEvPerLoop)
    eZeros = np.repeat(beamE, nEvPerLoop)
    eZeros -= lognorm.rvs(s=s, loc=eLoss, scale=scale, size=nEvPerLoop)
    checkForBadEs = True
    while checkForBadEs:
        badIdxs = np.where(eZeros <= 0.0)[0]
        nBads = badIdxs.shape[0]
        if nBads == 0:
            checkForBadEs = False
        replacements = np.repeat(beamE, nBads) - lognorm.rvs(s=s, loc=eLoss, scale=scale, size=nBads)
        eZeros[badIdxs] = replacements

#data_eD_matrix = odeint( dedxfxn, eZeros, x_binCenters )
    
    odesolver = ode( dedxForODE ).set_integrator('dopri5').set_initial_value(eZeros)
    for idx, xEvalPoint in enumerate(x_binCenters):
        sol = odesolver.integrate( xEvalPoint )
        #
        # STUFF FOR DEBUGGING
        #
        #print('shape of returned ode solution {}, first 10 entries {}'.format(sol.shape, sol[:10]))
        # data_eD_matrix = odesolver.integrate( x_binCenters )
            #print('shape of returned ode solution {}, first 10 entries {}'.format(data_eD_matrix.shape, data_eD_matrix[:10]))
        # data_eD = data_eD_matrix.flatten('K')
        #
        # END STUFF FOR DEBUGGING
        #
        data_weights = ddnXSfxn.evaluate(sol)
        hist, edEdges = np.histogram( sol, bins=eD_bins, range=(eD_minRange, eD_maxRange), weights=data_weights)
        dataHist[idx,:] += hist
        
        noWeightHist, edEdges = np.histogram( sol, bins=eD_bins, range=(eD_minRange, eD_maxRange))
        unweightedDataHist[idx,:] += noWeightHist

    solutions = np.zeros((1,x_bins))
    for eZero in eZeros:
        sol = interp_spline(eZero, x_binCenters)
        solutions = np.vstack((solutions, sol))
    print('shape of solutions {}'.format(solutions.shape))
    solutions = solutions[1:,:]
    eSolList = []
    eSolArray = np.zeros(eD_bins)
    #for xStepSol in solutions[:,]:
    #    data_weights = ddnXSfxn.evaluate(xStepSol)
    #    hist, edEdges = np.histogram(xStepSol, bins=eD_bins, range=(eD_minRange, eD_maxRange), weights=data_weights)
    #    eSolList.append(hist)
    for xStepSol in solutions.T:
        data_weights = ddnXSfxn.evaluate(xStepSol)
        hist, edEdges = np.histogram(xStepSol, bins=eD_bins, range=(eD_minRange, eD_maxRange), weights=data_weights)
        eSolList.append(hist)
        eSolArray = np.vstack((eSolArray,hist))
    eSolArray = eSolArray[1:,:]