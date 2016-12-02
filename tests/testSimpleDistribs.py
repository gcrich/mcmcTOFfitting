#!/Users/grayson/Dev/anaconda/python3/anaconda/bin/python

from pymc3 import Normal,HalfNormal,find_MAP,Model,traceplot,NUTS,sample
from pymc3 import Uniform, summary
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# these values are really not relevant here
# but i wrote them up, so stash them..
speedOfLight = 29.9792 # in cm/ns
massOfDeuteron = 1.8756e+06 # keV /c^2
massOfNeutron = 939565 # keV/c^2


distance_CellToZero = 518.055 # cm, distance from tip of gas cell to 0deg face
distance_cellLength = 2.86 # cm, length of gas cell
distance_zeroDegLength = 3.81 # cm, length of 0deg detector


# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 5000

x1_mean = 10
x1_sigma = 1

x2_mean = 15
x2_sigma = 3

# Predictor variable
X1 = np.random.normal(x1_mean,x1_sigma,size)
X2 = np.random.normal(x2_mean,x2_sigma,size)

print('size of X1 {} size of X2 {}'.format(len(X1), len(X2)) )

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma
print('length of observable Y data {}'.format(len(Y)) ) 

plt.figure(1)
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,4))
axes[0].scatter(X1, Y, alpha=0.25)
axes[1].scatter(X2, Y, alpha=0.25)
axes[0].set_ylabel('Y'); axes[0].set_xlabel('X1'); axes[1].set_xlabel('X2');
plt.show()


plt.figure(2)
# plot X vs X2
fig_x_v_x2 = plt.scatter(X1,X2,label='X1 vs X2')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()


# plot the actual observable Y distribution
hist_y, histBins_y = np.histogram(Y, bins=100)
plt.figure(3)
plt.hist(Y, 100, alpha=0.7, label='Distribution of observable Y')
plt.xlabel('Y')
plt.ylabel('Counts')
plt.show()


basic_model = Model()

with basic_model:
    
    # Priors for unknown model parameters
    alpha = Normal('alpha', mu=0, sd=10)
    beta = Normal('beta', mu=0, sd=10, shape=2)
    sigma = HalfNormal('sigma', sd=1)
    
    x1_mean = Uniform('x1_mean', lower=5, upper=15)
    x1_sigma = Uniform('x1_sigma', lower=0.5, upper=1.5 )
    x2_mean = Uniform('x2_mean', lower=10, upper =20)
    x2_sigma = Uniform('x2_sigma', lower=1, upper=6)
    
    
    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2
    
    # Likelihood (sampling distribution) of observations
    Y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

    map_estimate = find_MAP(model=basic_model)
#    map_estimate = find_MAP(model=basic_model, fmin=optimize.fmin_powell)
    
    trace=sample( 2000, start=map_estimate, njobs=4)
    
traceplot(trace)
summary(trace)