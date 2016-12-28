#
# test of ion stopping via bethe equation
#
# also tests implementation in ionStopping class
#

from __future__ import print_function
import numpy as np
from numpy import inf
import scipy.optimize as optimize
from scipy.integrate import odeint
import scipy.constants as sciConsts
import matplotlib.pyplot as plot
from constants.constants import (masses, qValues, distances, physics)
from utilities.utilities import ddnXSinterpolator
from utilities.ionStopping import ionStopping

def stoppingFxn(energy, x=0):
    """
    Calculate dE/dx according to Bethe formula
    x here is a dummy variable
    """
# make energy into velocity which goes into the bethe formula
    velocity =np.sqrt(2 * energy / masses.deuteron) * physics.speedOfLight
#    print('velocity for energy {} is {}'.format(energy,velocity))
    density = 8.37e-5
    mmc = 1 # in g/mol, molar mass constant
    numDensity = sciConsts.Avogadro * 1 * density / (2 * mmc) # n in equation
    ionCharge = 1 # little z in equation, in multiples of elementary charge
    leadingTerm = (4 * np.pi * numDensity * ionCharge**2 /
                   (masses.electron * physics.speedOfLight**2 * velocity**2))
    #squareTerm = (sciConsts.e**2/(4 * np.pi * physics.epsilon_0))**2
    squareTerm = 1.67489e-14 # worked out on paper
    iExcitationPot = 19.2 # from http://pdg.lbl.gov/2016/AtomicNuclearProperties/HTML/deuterium_gas.html
    logArg = (2 * masses.electron /(physics.speedOfLight**2)* velocity**2 /
              (iExcitationPot*1e-3))
    stopping = -1 * leadingTerm * squareTerm * np.log(logArg)
#    print('leading term {}\nmiddle term {}\nlog term {}'.format(np.log10(leadingTerm),
#                                                                squareTerm,
#                                                                np.log(logArg)))
    return stopping


stoppingModel = ionStopping.simpleBethe([1, 2, 8.37e-5, 1, 19.2])

energies = np.linspace(900, 1100, 3)
#stoppingPowers = stoppingFxn(energies)
print('D energy\tstopping from fxn\tstopping from class')
for idx, energy in enumerate(energies):
    print('{}\t{}\t{}'.format(energy,stoppingFxn(energy),stoppingModel.dEdx(energy)))


eZeros = np.random.normal(900.0, 50, 5000)

xLocations = np.linspace(0.0, distances.tunlSSA_CsI.cellLength, 50)
solutions = odeint( stoppingModel.dEdx, eZeros, xLocations)


for idx in range(0,solutions.shape[1]):
    plot.scatter(xLocations,solutions[:,idx],alpha=0.1,color='k')
plot.show()