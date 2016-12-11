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
import matplotlib.pyplot as plt
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
    N = len(y)
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
m_true=-0.15
b_true=6.3

sigma_bad = 0.4
m_bad = -0.2
b_bad = 6.2
    
nSamples = 50

xVals = np.random.uniform(low=xMin, high=xMax, size=nSamples)
yVals = np.random.normal(getShiftingMean(xVals,m_true,b_true), sigma_true)
yVals_bad = np.random.normal(getShiftingMean(xVals,m_bad,b_bad),sigma_bad)

plt.figure(1)
plt.scatter(xVals,yVals,color='green')
plt.scatter(xVals,yVals_bad,color='red',alpha=0.3)
plt.show()

plt.figure(2)
plt.hist(yVals,bins=100,color='green')
plt.hist(yVals_bad, bins=100,color='red',alpha=0.3)
plt.show()


fxnIntegral = quad(getProjectedProb, -20, 20, 
                   args=(m_true, b_true, sigma_true, xMin, xMax))
print('integral of projected function is {}'.format(fxnIntegral))
scaleFactor = nSamples / fxnIntegral[0]

pdfPlotX = np.linspace(-20,20,100)
pdfPlotY = getProjectedProb(pdfPlotX,m_true,b_true,sigma_true, xMin,xMax)
numericalPdfY = getNumProjectedProb(sigma_true, m_true, b_true, pdfPlotX)
plt.figure(3)
plt.scatter(pdfPlotX,pdfPlotY, color='green', alpha=0.3)
plt.scatter(pdfPlotX, numericalPdfY, color='k', alpha=0.3)
plt.show()

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

nllX_sig = np.linspace(0.05,1.0,100)
nllY_sig_num = []
for xVal in nllX_sig:
    nllY_sig_num.append( numericalNLL([xVal,m_true,b_true], yVals) )
plt.figure(4)
plt.scatter(nllX_sig,nllY_sig_num)
plt.xlabel('sigma')
plt.ylabel('NLL value')
plt.show()

nllX_b = np.linspace(0.,7.,70)
nllY_b = []
for xVal in nllX_b:
    nllY_b.append(numericalNLL([sigma_true,m_true,xVal],yVals))
plt.figure(5)
plt.scatter(nllX_b,nllY_b)
plt.xlabel('b (offset of shifting mean)')
plt.ylabel('NLL value')
plt.show()

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
ndim, nwalkers = 3, 100

sigma, m, b = [0.4, -0.3, 5]
pos = [[sigma, m, b] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
print(type(yVals))
print(len(yVals))
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobNumeric, 
                                kwargs={'observables': yVals}, threads = 4)
sampler.run_mcmc(pos, 500)


plt.figure(10)
plt.plot(sampler.chain[:,:,0].T,'-',alpha=0.3)
plt.ylabel('sigma')
plt.show()

plt.figure(11)
plt.plot(sampler.chain[:,:,1].T,'-',alpha=0.3)
plt.ylabel('m')
plt.show()

plt.figure(12)
plt.plot(sampler.chain[:,:,2].T,'-',alpha=0.3)
plt.ylabel('b')
plt.show()
#######################################
#
#
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
#
#
#
import corner
fig = corner.corner(samples, labels=["$sigma$", "$m$", "$b$"],
                      truths=[sigma_true, m_true, b_true])
#
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
