#
#
# simpleTOFfit.py
# created winter 2016 g.c.rich
#
# this is a scaled-back TOF FIT to test MCMC fitting
# doesn't include all the realistic features needed to do actual analysis
# BUT IT DOES FIT REAL DATA
# or at least try to
#
# TODO: add effective posterior predictive check (PPC)
#   that is, sample from the posteriors and produce some fake data
#   compare this fake data with observed data

import numpy as np
from numpy import inf
from scipy.integrate import quad
import scipy.optimize as optimize
import matplotlib.pyplot as plot
import emcee
import csv as csvlib

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
distance_tipToColli = 148.4 # cm, distance from cell tip to collimator exit
distance_colliToZero = 233.8 # cm, distance from collimator exit to face of
# 0deg AT ITS CLOSEST LOCATION
distance_delta1 = 131.09 # cm, difference between close and mid 0degree loc.
distance_delta2 = 52.39 # cm, difference between mid and far 0deg loc

# in function generateModelData the standoff distance has 0deg length / 2 added
distance_standoffClose= distance_tipToColli + distance_colliToZero
distance_standoffMid = distance_standoffClose + distance_delta1
distance_standoffFar = distance_standoffMid + distance_delta2

##############
# vars for binning of TOF 
tof_nBins = 25
tof_minRange = 180.0
tof_maxRange = 205.0
tof_range = (tof_minRange,tof_maxRange)


# PARAMETER BOUNDARIES
min_e0, max_e0 = 900,1100
min_e1,max_e1= -100, 0
min_e2,max_e2 = -30, 0
min_e3,max_e3 = -10, 0
min_sigma,max_sigma = 40, 100


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
    
    
def generateModelData(params, standoffDistance, nSamples):
    """Generate some fake data from our mode with num samples = nSamples
    params is an array of the parameters, [e0, e1, sigma]
    Returns a tuple of len(nSamples), [x, ed, en, tof]
    """
    initialEnergy, eLoss, e2, e3, sigma = params
    
    data_x=np.random.uniform(low=0.0, high=distance_cellLength, size=nSamples)
    meanEnergy = initialEnergy + eLoss*data_x + \
                              e2*np.power(data_x,2) + e3 * np.power(data_x,3)
    data_ed= np.random.normal(loc=meanEnergy, scale=sigma)
    data_en = getDDneutronEnergy(data_ed)
    
    neutronDistance = standoffDistance + (distance_cellLength - data_x) + \
                        distance_zeroDegLength/2
    neutronTOF = getTOF(mass_neutron, data_en, neutronDistance)
    effectiveDenergy = (initialEnergy + data_ed)/2
    deuteronTOF = getTOF( mass_deuteron, effectiveDenergy, data_x )
    data_tof = neutronTOF + deuteronTOF
    
    data = np.column_stack((data_x,data_ed,data_en,data_tof))
    return data
    
def lnlike(params, observables, nDraws=1000000):
    """Evaluate the log likelihood given a set of params and observables
    Observables is a vector; a histogrammed time distribution
    Params is a list of [initial D energy, E_D loss, sigma]
    nDraws is the number of points drawn from (energy,location) distribution\
    which are used to produce the PDF evaluations at different TOFs
    """
    #print('checking type ({}) and length ({}) of params in lnlikefxn'.format(type(params),len(params)))
    evalData=generateModelData(params, distance_standoffMid, nDraws)
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
    
    

def lnprior(theta):
    e_0, e_1, e_2, e_3, sigma = theta
    if 900 < e_0 < 1100 and -100 <e_1<0 and -30 < e_2 < 0 \
    and -10 < e_3 < 0 and 40 < sigma < 100:
        return 0
    return -inf
    
def lnprob(theta, observables):
    """Evaluate the log probability
    theta is a list of the model parameters
    observables is, in this case, a histogram of TOF values
    """
    prior = lnprior(theta)
    if not np.isfinite(prior):
        return -inf
    return prior + lnlike(theta, observables)
   
    
def readMultiStandoffTOFdata(filename):
    """Read in data from TAC for multiple-standoff runs
    Specify filename to access.
    """
    lowerBinEdges =[]
    tofCounts=[]
    with open(filename,'r') as tofFile:
        csvreader = csvlib.DictReader(tofFile, delimiter='\t', 
                                  fieldnames=['lowEdge','run0',
                                  'run1','run2','run3'])
        for row in csvreader:
            lowerBinEdges.append(float(row['lowEdge']))
            tofCounts.append([float(row['run0']), float(row['run1']),
                                    float(row['run2']), float(row['run3'])])
    tofData = np.column_stack((lowerBinEdges,tofCounts))
    return tofData
    
    
# mp_* are model parameters
# *_t are 'true' values that go into our fake data
# *_guess are guesses to start with
mp_e0_guess = 1050 # initial deuteron energy, in keV
mp_e1_guess = -80 # energy loss, 0th order approx, in keV/cm
mp_e2_guess = -10
mp_e3_guess = -5
mp_sigma_guess = 80 # width of deuteron energy spread, fixed for now, in keV


# get the data from file
tofData = readMultiStandoffTOFdata('/home/gcr/particleyShared/quenchingFactors/tunlCsI_Jan2016/data/CODA/data/multistandoff.dat')


binEdges = tofData[:,0]

#observedTOF, observed_bin_edges = np.histogram(fakeData[:,3], 
#                                               tof_nBins, tof_range)
observedTOF = tofData[:,1][(binEdges >= 180) & (binEdges < 205)]
observedTOFbinEdges = tofData[:,0][(binEdges>=180)&(binEdges<205)]



plot.figure()
plot.scatter(binEdges, tofData[:,1] )
plot.xlabel('tof (ns)')
plot.ylabel('counts')
plot.xlim(170.0,220.0)
plot.show()

# generate fake data
nSamples = 10000
fakeData = generateModelData([mp_e0_guess,mp_e1_guess,mp_e2_guess, 
                              mp_e3_guess, mp_sigma_guess], 
                              distance_standoffMid, nSamples)



# plot the fake data...
# but only 2000 points, no need to do more
plot.figure()
plot.scatter(fakeData[:2000,0], fakeData[:2000,2], color='k', alpha=0.3)
plot.xlabel('Cell location (cm)')
plot.ylabel('Neutron energy (keV)')
plot.show()


# plot the TOF 
plot.figure()
plot.subplot(211)
plot.hist(fakeData[:,3], 25, (180,205))
plot.ylabel('counts')
plot.subplot(212)
plot.scatter(observedTOFbinEdges,observedTOF)
plot.xlim(180,205)
plot.xlabel('TOF (ns)')
plot.ylabel('counts')
plot.show()

# plot the TOF vs x location
# again only plot 2000 points
plot.figure()
plot.scatter(fakeData[:2000,2],fakeData[:2000,3], color='k', alpha=0.3)
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

nll = lambda *args: -lnlike(*args)


testNLL = nll([mp_e0_guess, mp_e1_guess, mp_e2_guess, 
               mp_e3_guess, mp_sigma_guess], observedTOF)
print('test NLL has value {}'.format(testNLL))


parameterBounds=[(min_e0,max_e0),(min_e1,max_e1),(min_e2,max_e2),
                 (min_e3,max_e3),(min_sigma,max_sigma)]
minimizedNLL = optimize.minimize(nll, [mp_e0_guess,
                                       mp_e1_guess, mp_e2_guess, 
                                       mp_e3_guess, mp_sigma_guess], 
                                       args=observedTOF, method='TNC',
                                       tol=1.0,  bounds=parameterBounds)

print(minimizedNLL)


nDim, nWalkers = 5, 100

e0, e1, e2, e3, sigma = minimizedNLL["x"]

p0 = [[e0,e1,e2,e3,sigma] + 1e-1 * np.random.randn(nDim) for i in range(nWalkers)]
sampler = emcee.EnsembleSampler(nWalkers, nDim, lnprob, 
                                kwargs={'observables': observedTOF}, 
                                threads=10)

#sampler.run_mcmc(p0, 500)
# run with progress updates..
for i, samplerResult in enumerate(sampler.sample(p0, iterations=500)):
    if (i+1)%10 == 0:
        print("{0:5.1%}".format(float(i)/500))

plot.figure()
plot.subplot(511)
plot.plot(sampler.chain[:,:,0].T,'-',color='k',alpha=0.2)
plot.ylabel('Initial energy (keV)')
plot.subplot(512)
plot.plot(sampler.chain[:,:,1].T,'-',color='k',alpha=0.2)
plot.ylabel('Energy loss (keV/cm)')
plot.subplot(513)
plot.plot(sampler.chain[:,:,2].T,'-',color='k',alpha=0.2)
plot.ylabel(r'$E_2$ (keV/cm$^2$)')
plot.xlabel('Step')
plot.subplot(514)
plot.plot(sampler.chain[:,:,3].T,'-',color='k',alpha=0.2)
plot.ylabel(r'$E_3$ (keV/cm$^3$)')
plot.xlabel('Step')
plot.subplot(515)
plot.plot(sampler.chain[:,:,4].T,'-',color='k',alpha=0.2)
plot.ylabel('Sigma (keV)')
plot.xlabel('Step')
plot.show()


samples = sampler.chain[:,200:,:].reshape((-1,nDim))
import corner as corn
cornerFig = corn.corner(samples,labels=["$E_0$","$E_1$","$E_2$","$E_3$",
                                        "$\sigma$"],
                        quantiles=[0.16,0.5,0.84], show_titles=True,
                        title_kwargs={'fontsize': 12})