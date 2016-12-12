#
#
# simpleTOFmodel.py
# created winter 2016 g.c.rich
#
# this is a scaled-back TOF model to test MCMC fitting
# doesn't include all the realistic features needed to do actual analysis

import numpy as np
from scipy.integrate import quad
import scipy.optimize as optimize
import matplotlib.pyplot as plot
import emcee

#
# CONSTANTS
#
# these are perhaps presently not included in a very pythonic way
# but to get going, here we are..
#
speedOfLight = 29.9792 # in cm/ns
mass_deuteron = 1.8756e+06 # keV /c^2
mass_neutron = 939565.0 # keV/c^2
mass_he3 = 2.809414e6 # keV/c^2

# Q value of DDN reaction, in keV
qValue_ddn = 3268.914


distance_cellToZero = 518.055 # cm, distance from tip of gas cell to 0deg face
distance_cellLength = 2.86 # cm, length of gas cell
distance_zeroDegLength = 3.81 # cm, length of 0deg detector


def getDDneutronEnergy(deuteronEnergy, labAngle = 0):
    """Get the energy of neutrons produced by DDN reaction
    Function accepts the deuteron energy (in keV) and the angle (in lab\
    frame of reference) at which the neutron is emitted.
    Returns neutron energy in keV
    """     
    neutronAngle_radians = labAngle * np.pi / 180
    rVal = np.sqrt(mass_deuteron * mass_neutron*deuteronEnergy) / \
                   (mass_neutron + mass_he3) * \
                   np.cos(neutronAngle_radians)
    sVal = (deuteronEnergy *( mass_he3 - mass_deuteron) +
            qValue_ddn * mass_he3) / (mass_neutron + mass_he3)
    sqrtNeutronEnergy = rVal + np.sqrt(np.power(rVal,2) + sVal)
    return np.power(sqrtNeutronEnergy, 2)
    
def getTOF(mass, energy, distance):
    """Compute time of flight, in nanoseconds, given\
    mass of particle (in keV/c^2), the particle's energy (in keV),\
    and the distance traveled (in cm).
    Though simple enough to write inline, this will be used often.
    """
    velocity = speedOfLight * np.sqrt(2 * energy / mass)
    tof = distance / velocity
    return tof
    
    
def evaluateModel()
   
# mp_* are model parameters
# *_t are 'true' values that go into our fake data
mp_initialEnergy_t = 1100 # initial deuteron energy, in keV
mp_loss0_t = -100 # energy loss, 0th order approx, in keV/cm
mp_sigma_t = 50 # width of deuteron energy spread, fixed for now, in keV


# generate fake data
nSamples = 10000
data_model_x = np.random.uniform(low=0.0, high=distance_cellLength, 
                                 size=nSamples)
data_model_ed = np.random.normal(loc=mp_initialEnergy_t + 
                                 mp_loss0_t * data_model_x,
                                 scale=mp_sigma_t)
data_model_en = getDDneutronEnergy(data_model_ed, 0.0)

# we've got now a neutron energy distribution over the length of the cell
# let's make some TOF data from that
neutronDistance = distance_cellToZero + (distance_cellLength - data_model_x)
neutronTOF = getTOF(mass_neutron, data_model_en, neutronDistance)
effectiveDenergy = (mp_initialEnergy_t + data_model_ed)/2
deuteronTOF = getTOF( mass_deuteron, effectiveDenergy, data_model_x )
modelTOFdata = neutronTOF + deuteronTOF


# plot the fake data...
plot.figure(1)
plot.scatter(data_model_x, data_model_en, color='k', alpha=0.3)
plot.xlabel('Cell location (cm)')
plot.ylabel('Neutron energy (keV)')
plot.show()


# plot the TOF 
plot.figure(2)
plot.hist(modelTOFdata, bins=50)
plot.xlabel('TOF (ns)')
plot.show()

# plot the TOF vs x location
plot.figure(3)
plot.scatter(data_model_en,modelTOFdata, color='k', alpha=0.3)
plot.xlabel('Neutron energy (keV)' )
plot.ylabel('TOF (ns)')
plot.show()

##########################################
# here's where things are going to get interesting...
# in order to do MCMC, we are going to have to have a log probability fxn
# this means, we need a log LIKELIHOOD function, and this means we
# need just a regular old pdf
# unfortunately, even a regular old PDF is a hideously complicated thing
# no real chance of an analytical approach
# but we can NUMERICALLY attempt to do things
# for a given 