import numpy as np
import scipy.stats as spstats
from utilities.pdfs import skewnorm
import matplotlib.pyplot as plt

xVals = np.linspace(500, 1000, 1000)

plt.figure()
plt.plot(xVals, skewnorm.pdf(xVals, a=-1.5, scale=70, loc=770), ls='dashed', color='r')
plt.plot(xVals, spstats.skewnorm.pdf(xVals, a=-1.5, scale=70, loc=770), ls='dotted',color='g')
plt.draw()


customData = skewnorm.rvs(a=-1.5, loc=770, scale=70,size=5000)
realData = spstats.skewnorm.rvs(a=-1.5, loc=770, scale=70, size=5000)
plt.figure()
plt.hist( customData, alpha=0.3, color='r', bins=50, range=(500,1000))
plt.hist( realData, alpha=0.3, color='g', bins=50, range=(500,1000))
plt.draw()
plt.show()