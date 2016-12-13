#
#
# mcModelIntegration.py
# created winter 2016 g.c.rich
#
# intended to demo and develop MC-based integration of models
# needed to numerically evaluate likelihoods / probabilities
# 
# all the 'magic' here is in the functions lnlike and generateModelData

import numpy as np
from numpy import inf
from scipy.integrate import quad
import scipy.optimize as optimize
import matplotlib.pyplot as plot


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

##############
# vars for binning of TOF 
tof_nBins = 25
tof_minRange = 175.0
tof_maxRange = 200.0
tof_range = (tof_minRange,tof_maxRange)


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
    
    
def evaluateModel( params, coords):
    """Evaluate PDF at specific location in 2D model space
    Params are the model parameters (ED_0, ED_1, sigma)
    coords are the coordinates at which to evaluate
    """
    mp_initialEnergy, mp_loss0, mp_sigma = params
    x, energyDeuteron = coords
    mean = mp_initialEnergy + x * mp_loss0
    returnVal= np.exp(-1*np.power(energyDeuteron - mean,2)/(2*mp_sigma**2))
    return returnVal / (mp_sigma*np.sqrt(2*np.pi))
    
    
def generateModelData(params, nSamples):
    """Generate some fake data from our mode with num samples = nSamples
    params is an array of the parameters, [e0, e1, sigma]
    Returns a tuple of len(nSamples), [x, ed, en, tof]
    """
    initialEnergy, eLoss, sigma = params
    data_x=np.random.uniform(low=0.0, high=distance_cellLength, size=nSamples)
    data_ed= np.random.normal(loc=initialEnergy + eLoss*data_x, 
                              scale=sigma)
    data_en = getDDneutronEnergy(data_ed)
    
    neutronDistance = distance_cellToZero + (distance_cellLength - data_x)
    neutronTOF = getTOF(mass_neutron, data_en, neutronDistance)
    effectiveDenergy = (initialEnergy + data_ed)/2
    deuteronTOF = getTOF( mass_deuteron, effectiveDenergy, data_x )
    data_tof = neutronTOF + deuteronTOF
    
    data = np.column_stack((data_x,data_ed,data_en,data_tof))
    return data
    
def lnlike(params, observables, nDraws=2000000):
    """Evaluate the log likelihood given a set of params and observables
    Observables is a vector; a histogrammed time distribution
    Params is a list of [initial D energy, E_D loss, sigma]
    nDraws is the number of points drawn from (energy,location) distribution\
    which are used to produce the PDF evaluations at different TOFs
    """
    #print('checking type ({}) and length ({}) of params in lnlikefxn'.format(type(params),len(params)))
    evalData=generateModelData(params, nDraws)
    evalHist, evalBinEdges = np.histogram(evalData[:,3], tof_nBins, tof_range,
                                          density=True)
    logEvalHist = np.log(evalHist)
    #print(logEvalHist)
    # find what TOFs have zero observed data
    # we'll use this to handle cases where we might wind up with -inf*0
    # likelihood is fine if PDF is 0 somewhere where no data is found
    # without checks though, ln(PDF=0)=-inf, -inf*0 = nan
    # however, if PDF is 0 (lnPDF=-inf) where there IS data, lnL should be -inf
    zeroObservedIndices = np.where(observables == 0)[0]
    for idx in zeroObservedIndices:
        if logEvalHist[idx] == -inf:
            logEvalHist[zeroObservedIndices] = 0
    
    loglike = np.dot(logEvalHist,observables)
    return loglike
    
    
# mp_* are model parameters
# *_t are 'true' values that go into our fake data
mp_initialEnergy_t = 1100 # initial deuteron energy, in keV
mp_loss0_t = -100 # energy loss, 0th order approx, in keV/cm
mp_sigma_t = 50 # width of deuteron energy spread, fixed for now, in keV


# generate fake data
nSamples = 1000000
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
plot.scatter(data_model_x[:1000], data_model_en[:1000], color='k', alpha=0.3)
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
plot.scatter(data_model_en[:2000],modelTOFdata[:2000], color='k', alpha=0.3)
plot.xlabel('Neutron energy (keV)' )
plot.ylabel('TOF (ns)')
plot.show()

# just dump the data so we know what we're looking at
fakeDataSet = np.column_stack((data_model_x,data_model_ed,data_model_en,modelTOFdata))
#print(fakeDataSet)
# select the data for a given time of flight
tofSatisfiedData = fakeDataSet[np.ix_(np.floor(fakeDataSet[:,3])==185.0,(0,1,2))]
print('number of events found {}'.format(len(tofSatisfiedData)))

plot.figure(10)
plot.scatter(tofSatisfiedData[:500,0],tofSatisfiedData[:500,1],color='red',
             alpha=0.5, zorder=10)
plot.scatter(data_model_x[:5000], data_model_ed[:5000], color='k', alpha=0.2,zorder=1)
plot.ylabel('Deuteron energy (keV)')
plot.xlabel('Location in cell (cm)')
plot.show()

histTOFdata, tof_bin_edges = np.histogram(fakeDataSet[:,3], tof_nBins, 
                                          tof_range,
                                          density=True)
plot.figure(20)
plot.scatter(tof_bin_edges[:-1], histTOFdata, color='k')
plot.xlabel('TOF (ns)')
plot.ylabel('Counts')
plot.show()

plot.figure(21)
plot.scatter(tof_bin_edges[:-1],np.log(histTOFdata), color='k')
plot.xlabel('TOF (ns)')
plot.ylabel('log PDF')
plot.show()


# make some small "observed" fake data
nObsTestEvents = 5000
fakeObsData = generateModelData([mp_initialEnergy_t,mp_loss0_t,mp_sigma_t],
                                nObsTestEvents)


# make our vector of 'n'
observedVectorN, observed_bin_edges = np.histogram(fakeObsData[:,3], tof_nBins,
                                                   tof_range)


loghist = np.log(histTOFdata)
zeroIndices = np.where(observedVectorN==0)[0]
for idx in zeroIndices:
    if loghist[idx] == -inf:
        loghist[idx] = 0

manualLogLike = np.dot(loghist,observedVectorN)
print('manually computed loglike {}'.format(manualLogLike))

testLogLike = lnlike([mp_initialEnergy_t,mp_loss0_t,mp_sigma_t], 
                     observedVectorN)
testLogLikeBad = lnlike([mp_initialEnergy_t,mp_loss0_t,mp_sigma_t*0.8],
                        observedVectorN)
print('test loglikelihood value {}, and for off-observables {}'.format(
      testLogLike, testLogLikeBad))

# we're used to seeing NLLs, not the .. non-negative version
nll = lambda *args: -lnlike(*args)

testnll = nll([mp_initialEnergy_t,mp_loss0_t,mp_sigma_t], 
                     observedVectorN)
testnllbad = nll([mp_initialEnergy_t,mp_loss0_t,mp_sigma_t*0.8],
                        observedVectorN)
print('test NLL value {}, and for off-observables {}'.format(
      testnll, testnllbad))


# scan over a range of values for parameters
multiplier = np.linspace(0.5,1.5,20)
sigmaVals = mp_sigma_t*multiplier
initialVals = mp_initialEnergy_t * multiplier
lossVals = mp_loss0_t * multiplier
nll_sigmas =[]
nll_initials=[]
nll_losses=[]
print('scanning sigmas..')
for sig in sigmaVals:
    nll_sigmas.append(nll([mp_initialEnergy_t, mp_loss0_t, sig],
                 observedVectorN))
print('scanning initial energies..')
for initial in initialVals:
    nll_initials.append( nll([initial, mp_loss0_t, mp_sigma_t],
                  observedVectorN))
print('scanning losses...')
for loss in lossVals:
    nll_losses.append( nll([mp_initialEnergy_t, loss, mp_sigma_t],
               observedVectorN))

plot.figure()
plot.subplot(221)
plot.hist(fakeObsData[:,3], tof_nBins, tof_range)
plot.subplot(223)
plot.scatter(multiplier,nll_sigmas)
plot.xlabel('fraction sigma TRUE')
plot.ylabel('nll')
plot.subplot(222)
plot.scatter(multiplier,nll_initials)
plot.xlabel('fraction of TRUE initial energy')
plot.ylabel('NLL')
plot.subplot(224)
plot.scatter(multiplier, nll_losses)
plot.xlabel('fraction of TRUE energy loss parameter')
plot.ylabel('NLL')
plot.show()