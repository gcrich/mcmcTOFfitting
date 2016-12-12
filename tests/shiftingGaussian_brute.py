#
#
# shiftingGaussian_brute
# created winter 2016, g.c.rich
#
# this is a 'brute force' version of the shifting gaussian 
# basically we numerically integrate things \
# rather than working them out analytically
#
# the 'shifting gaussian' is a gaussian in Y whose mean is given by 
#       mu = m*x + b
# where m and b are parameters and X is a dimension we will marginalize over
#
# 
import numpy as np
from scipy.special import erf
from scipy.integrate import quad
import scipy.optimize as optimize
import matplotlib.pyplot as plot
import emcee

xMax = 10.0
xMin = 0.0

def getShiftingMean(x, m, b):
    return b + m * x

def evalPdf(x, y, sigma, m, b):
    """Evaluate the PDF including x and y dimensions
    """
    mean = m*x + b

    retVal = np.exp(-1 * np.power(y - mean,2)/(2*sigma*sigma))/(sigma*np.sqrt(2*np.pi)) 

    return retVal
    
def getProjectedProb(y, m, b, sigma, xMin, xMax):
    """Get the projected probability analytically
    I am suspicious that this isn't properly worked out
    Thought it qualitatively appears to work in some cases, it has a 1/m factor
    this makes it undefined at m=0, though it SHOULD be defined in this case
    """
    erfargMax = (b + m * xMax - y)/(np.sqrt(2)*sigma)
    erfargMin = (b + m * xMin - y)/(np.sqrt(2)*sigma)
    
    numerator =  np.sqrt(np.pi/2)*sigma*(erf(erfargMax) - erf(erfargMin))
    return numerator/m
    
def getNumProjectedProb(sigma, m, b, y):
    """Get the projected probability through numerical integration
    """
    if type(y) is float:
        projPdfVal, err = quad(evalPdf, xMin, xMax, args=(y, sigma, m, b))
        return projPdfVal
    else: # assume it is an array
        returnArray = []
        for yVal in y:
            projPdfVal, err = quad(evalPdf, xMin,xMax, args=(yVal,sigma,m,b))
            returnArray.append(projPdfVal)
        return returnArray
    
def lnlikesub(x, m, b, sigma, y):
    N = len(y)
    dLL = -N * np.log(sigma) - 1/(2*sigma*sigma) * np.sum(b + m*x - y)
    return dLL
    
    
    
def lnlike(sigma, m, b, observable):
    """Get the log likelihood 
    This is computed via quad integration of a subroutine over the parameter x
    """
    #xMax=10
    #xMin=-10
    #sigma, m, b = modelParams
    #print('sigma {}, m {}, b {}'.format(sigma,m,b))
    y = observable # for convenience inside the function
    #print('number of y vals {}'.format(N))
    loglikelihood, err = quad(lnlikesub,xMin,xMax, args=(m, b, sigma, y) )
    return loglikelihood
    
def lnlikeFromProjProb(modelParams, observable):
    """Get the log likelihood from the projected PDF
    argument modelParams is an array of the model parameters, [sigma, m, b]
    argument observables is an array of data to fit against
    """
    #xMax=10
    #xMin=-10
    sigma, m, b = modelParams
    #print('sigma {}, m {}, b {}'.format(sigma,m,b))
    y = observable # for convenience inside the function
    projectedProbabilities = getProjectedProb( y, m, b, sigma, xMin, xMax)
    logProjProbs = np.log(projectedProbabilities)
    loglikelihood = np.sum(logProjProbs)
    
    return loglikelihood
    
def numLnlikeFromProjProb(modelParams, observable):
    """Get the log likelihood from the numerically projected PDF
    argument modelParams is an array of the model parameters, [sigma, m, b]
    argument observables is an array of data to fit against
    """
    #xMax=10
    #xMin=-10
    sigma, m, b = modelParams
    #print('sigma {}, m {}, b {}'.format(sigma,m,b))
    y = observable # for convenience inside the function
    projectedProbabilities = getNumProjectedProb( sigma, m, b, y)
    logProjProbs = np.log(projectedProbabilities)
    loglikelihood = np.sum(logProjProbs)
    
    return loglikelihood

def lnPriors(modelParams):
    """Get the priors for the model parameters
    These are just uniform priors right now, so this returns -np.inf for \
    anything out of the allowed range
    """
    sigma, m, b = modelParams
    if sigma > 0 and sigma < 5 and b > 0 and b < 10 and m<0.1 and m>-0.5:
        return 0
    else:
        return -np.inf


def lnprobProj(modelParams, observables):
    """Compute the log probability based on the projected PDF 
    argument modelParams is an array of the model parameters, [sigma, m, b]
    argument observables is an array of data to fit against
    """
    logprior = lnPriors(modelParams)
    if not np.isfinite(logprior):
        return -np.inf
    return logprior + lnlikeFromProjProb(modelParams, observables)
    
def lnprobNumeric(modelParams, observables):
    """Compute the log probability based on the NUMERICALLY projected PDF 
    argument modelParams is an array of the model parameters, [sigma, m, b]
    argument observables is an array of data to fit against
    """
    logprior = lnPriors(modelParams)
    if not np.isfinite(logprior):
        return -np.inf
    return logprior + numLnlikeFromProjProb(modelParams, observables)
    


  
sigma_true = 0.5
m_true=0.0
b_true=6.3

sigma_bad = 0.4
m_bad = -0.2
b_bad = 6.2
    
nSamples = 500

xVals = np.random.uniform(low=xMin, high=xMax, size=nSamples)
yVals = np.random.normal(getShiftingMean(xVals,m_true,b_true), sigma_true)
yVals_bad = np.random.normal(getShiftingMean(xVals,m_bad,b_bad),sigma_bad)

plot.figure(1)
plot.scatter(xVals,yVals,color='green')
plot.scatter(xVals,yVals_bad,color='red',alpha=0.3)
plot.show()

plot.figure(2)
plot.hist(yVals,bins=100,color='green')
plot.hist(yVals_bad, bins=100,color='red',alpha=0.3)
plot.show()


fxnIntegral = quad(getProjectedProb, -20, 20, 
                   args=(m_true, b_true, sigma_true, xMin, xMax))
print('integral of projected function is {}'.format(fxnIntegral))
scaleFactor = nSamples / fxnIntegral[0]

pdfPlotX = np.linspace(-20,20,100)
pdfPlotY = getProjectedProb(pdfPlotX,m_true,b_true,sigma_true, xMin,xMax)
numericalPdfY = getNumProjectedProb(sigma_true, m_true, b_true, pdfPlotX)
plot.figure(3)
plot.scatter(pdfPlotX,pdfPlotY, color='green', alpha=0.3)
plot.scatter(pdfPlotX, numericalPdfY, color='k', alpha=0.3)
plot.show()

testPoint1 = 6
testPoint2 = 0
print('value at {}: {}'.format(testPoint1,
      getProjectedProb(testPoint1,m_true,b_true,sigma_true,xMin,xMax)))
print('value at {}: {}'.format(testPoint2,
      getProjectedProb(testPoint2,m_true,b_true,sigma_true,xMin,xMax)))

projectedProbabilities = getProjectedProb( yVals, m_true, b_true, 
                                          sigma_true,xMin,xMax)
logProjProbs = np.log(projectedProbabilities)
projectedProbabilities_bad = getProjectedProb( yVals_bad, m_true, 
                                              b_true, sigma_true,xMin,xMax)
logProjProbs_bad = np.log(projectedProbabilities_bad)

print('length of yVals array {} and array of probabilities {}'.format(
      len(yVals), len(projectedProbabilities)))


prodProb = np.prod(projectedProbabilities)
prodProb_bad = np.prod(projectedProbabilities_bad)

sumLogProjProbs = np.sum(logProjProbs)
sumLogProjProbs_bad = np.sum(logProjProbs_bad)

logProjProbLike = lnlikeFromProjProb( [sigma_true, m_true, b_true], yVals)
logProjProbLike_bad = lnlikeFromProjProb( [sigma_bad, m_bad, b_bad], yVals)

print('manually computed likelihood for good params {} and bad params {}'\
      .format(prodProb, prodProb_bad))
print('manually computed LOG likelihood for good params {} and bad params {}'\
      .format(sumLogProjProbs, sumLogProjProbs_bad))
print('function computed LOG likelihood for good params {} and bad params {}'\
      .format(logProjProbLike, logProjProbLike_bad))
# Generate some synthetic data from the model.
#testLNlike = lnlike([sigma_true, m_true, b_true], np.asarray([0,-10,10,5,0,2,-2]))
testLNlike = lnlike(sigma_true, m_true, b_true, yVals)
print('likelihood fxn at right points {}'.format(testLNlike))

testLNlike = lnlike(sigma_bad, m_bad, b_bad, yVals)
print('likelihood fxn at slightly off points {}'.format(testLNlike))



nll = lambda *args: -lnlikeFromProjProb(*args)
nllRight = nll([sigma_true,m_true,b_true],yVals)
nllRight2 = nll([sigma_true,m_true,b_true],yVals)
print('NLL value with right params {}, and again {}'.format(nllRight, nllRight2))

numericalNLL = lambda *args: -numLnlikeFromProjProb(*args)
numericNLLtrue = numericalNLL( [sigma_true, m_true, b_true], yVals)
print('NLL value from numerical approach {}'.format(numericNLLtrue))
print('NLL numeric, slightly bad values {}'.format(
      numericalNLL([sigma_bad, m_bad, b_bad], yVals)))

nllX_sig = np.linspace(np.exp(0.05),np.exp(1.0),100)
nllY_sig_num = []
for xVal in nllX_sig:
    nllY_sig_num.append( numericalNLL([np.log(xVal),m_true,b_true], yVals) )
plot.figure(4)
plot.scatter(nllX_sig,nllY_sig_num)
plot.xlabel('sigma')
plot.ylabel('NLL value')
plot.show()

nllX_b = np.linspace(0.,7.,70)
nllY_b = []
for xVal in nllX_b:
    nllY_b.append(numericalNLL([sigma_true,m_true,xVal],yVals))
plot.figure(5)
plot.scatter(nllX_b,nllY_b)
plot.xlabel('b (offset of shifting mean)')
plot.ylabel('NLL value')
plot.show()

optimizeResult = optimize.minimize(numericalNLL, [sigma_true,m_true,b_true], args=yVals)
print(optimizeResult)

#optimizeResultBadStart = optimize.minimize(nll, [sigma_bad,m_bad,b_bad], args=yVals)
#print(optimizeResultBadStart)

## covariance determination stuff - not obviously needed?
## everything made here SEEMS to go unused
#A = np.vstack((np.ones_like(x), x)).T
#C = np.diag(yerr * yerr)
#cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
#b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))
#
#   
#    
#
#nll = lambda *args: -lnlike(*args)
#result = op.minimize(nll, [m_true, b_true, np.log(f_true)], args=(x, y, yerr))
#m_ml, b_ml, lnf_ml = result["x"]
#
#
#def lnprior(theta):
#    m, b, lnf = theta
#    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
#        return 0.0
#    return -np.inf
#    
#    
#def lnprob(theta, x, y, yerr):
#    lp = lnprior(theta)
#    if not np.isfinite(lp):
#        return -np.inf
#    return lp + lnlike(theta, x, y, yerr)
#    
#######################################
ndim, nwalkers = 3, 250

sigma, m, b = [0.4, -0.3, 5]
pos = [[sigma, m, b] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
print(type(yVals))
print(len(yVals))
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobNumeric, 
                                kwargs={'observables': yVals}, threads = 8)
sampler.run_mcmc(pos, 1000)


plot.figure(10)
plot.plot(sampler.chain[:,:,0].T,'-',alpha=0.3)
plot.ylabel('sigma')
plot.show()

plot.figure(11)
plot.plot(sampler.chain[:,:,1].T,'-',alpha=0.3)
plot.ylabel('m')
plot.show()

plot.figure(12)
plot.plot(sampler.chain[:,:,2].T,'-',alpha=0.3)
plot.ylabel('b')
plot.show()
#######################################
#
# let's print some diagnostics
#autocorrelations = sampler.acor
#print('autocorrelation time for sigma {}, m {}, b {}'.format(
#      autocorrelations[0], autocorrelations[1], autocorrelations[2]) )


acceptanceFractions = sampler.acceptance_fraction
plot.figure(21)
plot.hist(acceptanceFractions, bins=50)
plot.xlabel('acceptance fraction')
plot.ylabel('counts')
plot.show()
#
samples = sampler.chain[:, -200:, :].reshape((-1, ndim))
#
#
#
import corner
fig = corner.corner(samples, labels=["$sigma$", "$m$", "$b$"],
                      truths=[sigma_true, m_true, b_true])





#####
# NOW RUN A PARALLEL TEMPERING MCMC
nPTtemps = 20
nPTwalkers = 100
ptSampler = emcee.PTSampler( nPTtemps, nPTwalkers, ndim, 
                      numLnlikeFromProjProb, lnPriors)
#
p0 = optimizeResult["x"] + 1e-3*np.random.randn(nPTtemps,nPTwalkers,ndim)
for p, lnl, lnp in ptSampler.sample(p0,iterations=1000, threads=10):
    pass
ptSampler.reset()
for p, lnl, lnp in ptSampler.sample( p, lnprob0=lnp, lnlike0=lnl, 
                                    iterations=10000,thin=10, threads=10):
    pass


zeroTempChain = ptSampler.chain[0,...]
zeroTempSamples = zeroTempChain.reshape((-1,ndim))

plot.figure(50)
plot.plot(zeroTempChain[:,:,0].T,'-',alpha=0.3)
plot.ylabel('sigma')
plot.show()

plot.figure(51)
plot.plot(zeroTempChain[:,:,1].T,'-',alpha=0.3)
plot.ylabel('m')
plot.show()

plot.figure(52)
plot.plot(zeroTempChain[:,:,2].T,'-',alpha=0.3)
plot.ylabel('b')
plot.show()


fig = corner.corner(zeroTempSamples, labels=["$sigma$", "$m$", "$b$"],
                      truths=[sigma_true, m_true, b_true])
#
#import matplotlib.pyplot as pl
#xl = np.array([0, 10])
#pl.figure(1)
#for m, b, lnf in samples[np.random.randint(len(samples), size=100)]:
#    pl.plot(xl, m*xl+b, color="k", alpha=0.1)
#pl.plot(xl, m_true*xl+b_true, color="r", lw=2, alpha=0.8)
#pl.errorbar(x, y, yerr=yerr, fmt=".k")
#pl.show()
#
#
#samples[:, 2] = np.exp(samples[:, 2])
#m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#                             zip(*np.percentile(samples, [16, 50, 84],
#                                                axis=0)))
#
