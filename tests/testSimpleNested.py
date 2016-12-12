#!/Users/grayson/Dev/anaconda/python3/anaconda/bin/python

from pymc3 import Normal,HalfNormal,find_MAP,Model,traceplot,NUTS,sample
from pymc3 import Uniform, summary, Poisson
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# these values are really not relevant here
# but i wrote them up, so stash them..
speedOfLight = 29.9792 # in cm/ns
massOfDeuteron = 1.8756e+06 # keV /c^2
massOfNeutron = 939565.0 # keV/c^2
massOfHelium3 = 2.809414e6 # keV/c^2

# Q value of DDN reaction, in keV
qValue_ddn = 3268.914


distance_cellToZero = 518.055 # cm, distance from tip of gas cell to 0deg face
distance_cellLength = 2.86 # cm, length of gas cell
distance_zeroDegLength = 3.81 # cm, length of 0deg detector


def getDTOF(initialEnergy, finalEnergy, location):
    averageEnergy = (initialEnergy + finalEnergy)/2
    velocity = speedOfLight * np.sqrt(2 * averageEnergy / massOfDeuteron)
    tof = location / velocity
    return tof

def getTOF(energy, mass, distance):
    velocity = speedOfLight * np.sqrt(2 * energy / mass)
    tof = distance / velocity
    return tof

# getDDneutronEnergy
#
# pass in deuteron energy and lab emission angle (in degrees)
#
def getDDneutronEnergy(deuteronEnergy, labAngle = 0):
    '''
    //    borrow naming convention from iliadis
     //    sqrt of energy is given by r +/- sqrt(r^2 + s)
     //    for this reaction, we only take the +
     
     Double_t neutronLabAngle_radians = TMath::Pi() * labAngle / 180;
     
     
     Double_t rVal = TMath::Sqrt(mass_deuteron_amu * mass_neutron_amu * deuteronEnergy) /(mass_neutron_amu + mass_helium3_amu) * TMath::Cos( neutronLabAngle_radians );
     
     Double_t sVal = (deuteronEnergy * (mass_helium3_amu - mass_deuteron_amu) + qValue_DDn_keV * mass_helium3_amu) / (mass_neutron_amu + mass_helium3_amu);
     
     Double_t sqrtNeutronEnergy_keV = rVal + TMath::Sqrt( TMath::Power( rVal, 2 ) + sVal );
     
     return TMath::Power( sqrtNeutronEnergy_keV, 2 );
     '''
    neutronAngle_radians = labAngle * np.pi / 180
    rVal = np.sqrt(massOfDeuteron * massOfNeutron*deuteronEnergy) / \
                   (massOfNeutron + massOfHelium3) * \
                   np.cos(neutronAngle_radians)
    sVal = (deuteronEnergy *( massOfHelium3 - massOfDeuteron) +
            qValue_ddn * massOfHelium3) / (massOfNeutron + massOfHelium3)
    sqrtNeutronEnergy = rVal + np.sqrt(np.power(rVal,2) + sVal)
    return np.power(sqrtNeutronEnergy, 2)
    


# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 1.0, 1.0
beta = [1.0, 2.5]

# Size of dataset
size = 5000


# gas cell length
length_cell = 2.81

# parameters defining the deuteron energy
eD_params = [ 1000.0, -30.0, -1.0, -1.0 ]
eD_sigma = 50.0 # width of the D energy spread


locationInCell = np.random.uniform(low=0, high=length_cell, size=size)

# Predictor variable
energy_deuteron_mean = (eD_params[0] + locationInCell*eD_params[1] + 
                        eD_params[2] * locationInCell*locationInCell +
                        eD_params[3] * np.power(locationInCell,3) )
# simulate our distribution
energy_deuteron = np.random.normal(energy_deuteron_mean, eD_sigma)

print('size of deuteron energy array {}'.format(len(energy_deuteron) ) )

#==============================================================================
#  NOTE THAT THIS IS A LAZY APPROXIMATION OF DEUTERON TOF
#  it makes a quick and rough approximation of the effect of energy loss 
#==============================================================================
deuteronTOF = getDTOF(energy_deuteron, eD_params[0], 
                     locationInCell)



energy_neutron = getDDneutronEnergy(energy_deuteron)



#==============================================================================
# 
# CALCULATE NEUTRON TIME OF FLIGHT
#
#==============================================================================
neutronTOF = getTOF(energy_neutron, 
                    massOfNeutron,
                    distance_cellToZero + distance_cellLength - locationInCell)

#==============================================================================
# total TOF calculation
#==============================================================================
totalTOF = neutronTOF + deuteronTOF

plt.figure(1)
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,4))
axes[0].scatter(energy_deuteron, locationInCell, alpha=0.25)
axes[1].scatter(energy_deuteron_mean, locationInCell, alpha=0.25)
axes[0].set_ylabel('Y'); axes[0].set_xlabel('X1'); axes[1].set_xlabel('X2');
plt.show()


plt.figure(2)
plt.scatter(locationInCell, energy_neutron, alpha=0.25,
            label='Neutron energy distribution along cell')
plt.xlabel('Location in cell (cm)')
plt.ylabel('Neutron energy (keV)')
plt.show()
'''
plt.figure(1)
# plot X vs X2
plt.scatter(inputX,Y,label='inputX vs Y', alpha=0.3)
plt.xlabel('inputX')
plt.ylabel('Y')
plt.show()
'''

# plot the actual observable Y distribution
hist_y, histBins_y = np.histogram(locationInCell, bins=100)
plt.figure(3)
plt.hist(locationInCell, 100, alpha=0.7,
         label='Distribution along cell length')
plt.xlabel('Y')
plt.ylabel('Counts')
plt.show()

# plot the TOF associated with the deuteron transit
plt.figure(4)
plt.hist(deuteronTOF, 100, alpha=0.5, label='Deuteron TOF')
plt.xlabel('Deuteron time-of-flight')
plt.ylabel('Counts')
plt.show()

# plot the neutron energy distribution
plt.figure(5)
plt.hist(energy_neutron, 100, alpha=0.5, label='Neutron energy distribution')
plt.xlabel('Neutron energy (keV)')
plt.ylabel('Counts')
plt.show()

# plot the TOF 
plt.figure(6)
plt.hist(totalTOF, 100, alpha=0.5, label='Total TOF')
plt.xlabel('Time-of-flight')
plt.ylabel('Counts')
plt.show()



basic_model = Model()

with basic_model:

    # these are prior distributions of parameters
    #eD_params[0] = Uniform('eD_param0', lower=800.0, upper=1200.0)
    #eD_params[1] = Uniform('eD_param1', lower=-100.0, upper=0.0 )
    #eD_params[2] = Uniform('eD_param2', lower=-10.0, upper=0.0)
    eD_sigma = Uniform('eD_sigma', lower=20.0, upper =100.0)
    eD_params[0] = Uniform('eD_param0', lower=800.0, upper=1500.0)
    eD_params[1] = Uniform('eD_param1', lower=-200.0, upper=0.0)
    eD_params[2] = Uniform('eD_param2', lower=-20.0, upper=0.0)
    eD_params[3] = Uniform('eD_param3', lower=-10.0, upper=0.0)
    
    #cellLocation = pm.Deterministic(name='cellLocation', var=
    #                                pm.Uniform('cellLocDist',lower=0.0, 
    #                                           upper=distance_cellLength))
    #cellLocationDist = Uniform('cellLocationDist',
    #                           lower=0, upper=distance_cellLength)
    #cellLocation = cellLocationDist.random()
    cellLocation = np.random.uniform(low=0.0, high=distance_cellLength)
    
    # Expected value of outcome
    ed_mean = (eD_params[0] + cellLocation*eD_params[1] + 
                        eD_params[2] * np.power(cellLocation,2) +
                        eD_params[3] * np.power(cellLocation,3) )
    en_mean = getDDneutronEnergy(ed_mean)
    
    # Likelihood (sampling distribution) of observations
    Y_obs = Normal('energy_neutron', mu=en_mean, sd=eD_sigma, 
                   observed=energy_neutron)

 #   map_estimate = find_MAP(model=basic_model)
    map_estimate = find_MAP(model=basic_model, fmin=optimize.fmin_powell)

    #step = pm.NUTS(state=map_estimate)
    step = pm.Metropolis(state=map_estimate, vars=[eD_params[0],eD_params[1],eD_params[2], eD_params[3], eD_sigma]) # Instantiate MCMC sampling algorithm    
    #step = pm.HamiltonianMC(state=map_estimate, vars=[eD_params[0],eD_params[1],eD_params[2], eD_params[3], eD_sigma])
    #trace=sample( 7000, step, start=map_estimate, njobs=4)
    trace=sample( 100000, step, start=map_estimate, njobs=8)
    
traceplot(trace[-2000:])
summary(trace[-2000:])
#traceplot(trace[:])
#summary(trace[:])