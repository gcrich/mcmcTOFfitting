#
#
# simpleTOFmodel.py
# created winter 2016 g.c.rich
#
# this is a scaled-back TOF model to test MCMC fitting
# doesn't include all the realistic features needed to do actual analysis
#
# TODO: add effective posterior predictive check (PPC)
#   that is, sample from the posteriors and produce some fake data
#   compare this fake data with observed data

import numpy as np
from numpy import inf
from scipy.integrate import quad
import scipy.optimize as optimize
#import matplotlib.pyplot as plot
import emcee
from constants.constants import (masses, qValues, distances, physics)



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
    rVal = np.sqrt(masses.deuteron * masses.neutron*deuteronEnergy) / \
                   (masses.neutron + masses.he3) * \
                   np.cos(neutronAngle_radians)
    sVal = (deuteronEnergy *( masses.he3 - masses.deuteron) +
            qValues.ddn * masses.he3) / (masses.neutron + masses.he3)
    sqrtNeutronEnergy = rVal + np.sqrt(np.power(rVal,2) + sVal)
    return np.power(sqrtNeutronEnergy, 2)
    
def getTOF(mass, energy, distance):
    """Compute time of flight, in nanoseconds, given\
    mass of particle (in keV/c^2), the particle's energy (in keV),\
    and the distance traveled (in cm).
    Though simple enough to write inline, this will be used often.
    """
    velocity = physics.speedOfLight * np.sqrt(2 * energy / mass)
    tof = distance / velocity
    return tof
    
    
def generateModelData(params, nSamples):
    """Generate some fake data from our mode with num samples = nSamples
    params is an array of the parameters, [e0, e1, sigma]
    Returns a tuple of len(nSamples), [x, ed, en, tof]
    """
    initialEnergy, eLoss, sigma = params
    data_x=np.random.uniform(low=0.0, high=distances.tunlSSA_CsI.cellLength, 
                             size=nSamples)
    data_ed= np.random.normal(loc=initialEnergy + eLoss*data_x, 
                              scale=sigma)
    data_en = getDDneutronEnergy(data_ed)
    
    neutronDistance = distances.tunlSSA_CsI.cellToZero + (distances.tunlSSA_CsI.cellLength - data_x)
    neutronTOF = getTOF(masses.neutron, data_en, neutronDistance)
    effectiveDenergy = (initialEnergy + data_ed)/2
    deuteronTOF = getTOF( masses.deuteron, effectiveDenergy, data_x )
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
    
    

def lnprior(theta):
    e_0, e_1, sigma = theta
    if 800 < e_0 < 1200 and -200 <e_1<0 and 10 < sigma < 100:
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
   
# mp_* are model parameters
# *_t are 'true' values that go into our fake data
mp_initialEnergy_t = 1100 # initial deuteron energy, in keV
mp_loss0_t = -100 # energy loss, 0th order approx, in keV/cm
mp_sigma_t = 50 # width of deuteron energy spread, fixed for now, in keV


# generate fake data
nSamples = 10000
fakeData = generateModelData([mp_initialEnergy_t,mp_loss0_t,mp_sigma_t],
                             nSamples)



## plot the fake data...
#plot.figure(1)
#plot.scatter(fakeData[:,0], fakeData[:,2], color='k', alpha=0.3)
#plot.xlabel('Cell location (cm)')
#plot.ylabel('Neutron energy (keV)')
#plot.show()
#
#
## plot the TOF 
#plot.figure(2)
#plot.hist(fakeData[:,3], bins=50)
#plot.xlabel('TOF (ns)')
#plot.show()
#
## plot the TOF vs x location
#plot.figure(3)
#plot.scatter(fakeData[:,2],fakeData[:,3], color='k', alpha=0.3)
#plot.xlabel('Neutron energy (keV)' )
#plot.ylabel('TOF (ns)')
#plot.show()

##########################################
# here's where things are going to get interesting...
# in order to do MCMC, we are going to have to have a log probability fxn
# this means, we need a log LIKELIHOOD function, and this means we
# need just a regular old pdf
# unfortunately, even a regular old PDF is a hideously complicated thing
# no real chance of an analytical approach
# but we can NUMERICALLY attempt to do things

nll = lambda *args: -lnlike(*args)


observedTOF, observed_bin_edges = np.histogram(fakeData[:,3], 
                                               tof_nBins, tof_range)

#minimizedNLL = optimize.minimize(nll, [1080,
#                                       mp_loss0_t *1.2,mp_sigma_t *1.05], 
#                                       args=observedTOF, method='Nelder-Mead',
#                                       tol=1.0)
#
#print(minimizedNLL)


nDim, nWalkers = 3, 50

#e0, e1, sigma = minimizedNLL["x"]
e0, e1, sigma = mp_initialEnergy_t*1.01, mp_loss0_t*1.1, mp_sigma_t * 0.8
print("lnlike at initial guess is {}".format(lnlike([e0,e1,sigma], observedTOF)))

p0 = [[e0,e1,sigma] + 1e-2 * np.random.randn(nDim) for i in range(nWalkers)]
sampler = emcee.EnsembleSampler(nWalkers, nDim, lnprob, 
                                kwargs={'observables': observedTOF}, 
                                threads=8)

#sampler.run_mcmc(p0, 500)
# run with progress updates..
for i, samplerResult in enumerate(sampler.sample(p0, iterations=500)):
    if (i+1)%2 == 0:
        print("{0:5.1%}".format(float(i)/500))


samples = sampler.chain[:,300:,].reshape((-1,nDim))
# Compute the quantiles.
# this comes from https://github.com/dfm/emcee/blob/master/examples/line.py
e0_mcmc, e1_mcmc, sigma_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
print("""MCMC result:
    E0 = {0[0]} +{0[1]} -{0[2]} (truth: {1})
    E1 = {2[0]} +{2[1]} -{2[2]} (truth: {3})
    sigma = {4[0]} +{4[1]} -{4[2]} (truth: {5})
    """.format(e0_mcmc, mp_initialEnergy_t, e1_mcmc, mp_loss0_t,
               sigma_mcmc, mp_sigma_t))
        
        
        
#plot.figure()
#plot.subplot(311)
#plot.plot(sampler.chain[:,:,0].T,'-',color='k',alpha=0.2)
#plot.ylabel('Initial energy (keV)')
#plot.subplot(312)
#plot.plot(sampler.chain[:,:,1].T,'-',color='k',alpha=0.2)
#plot.ylabel('Energy loss (keV/cm)')
#plot.subplot(313)
#plot.plot(sampler.chain[:,:,2].T,'-',color='k',alpha=0.2)
#plot.ylabel('Sigma (keV)')
#plot.xlabel('Step')
#plot.show()


#samples = sampler.chain[:,200:,:].reshape((-1,nDim))
#import corner as corn
#cornerFig = corn.corner(samples,labels=["$E_0$","$E_1$","$\sigma$"],
#                        truths=[mp_initialEnergy_t, mp_loss0_t, mp_sigma_t],
#                        quantiles=[0.16,0.5,0.84], show_titles=True,
#                        title_kwargs={'fontsize': 12})