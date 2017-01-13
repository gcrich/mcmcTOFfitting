#
# ppcTools.py
#
# part of mcmcTOFfitting
# this is a set of utilities that assist in the sampling of a posterior distr.
# this aides in producing a 'posterior predictive check' or ppc
#
# armed with a posterior we get from MCMC runs, we want to produce some fake
# data using the posterior PDF
# we can then compare this with the input data and see how things look

from __future__ import print_function
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from constants.constants import (masses, distances, physics)
from utilities.utilities import (beamTimingShape, ddnXSinterpolator,
                                 getDDneutronEnergy)
from utilities.ionStopping import ionStopping

class ppcTools:
    """Assist in the sampling of posterior distributions from MCMC TOF fits
    """