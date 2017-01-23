#
# obnoxious redefinition of PDF(s) because cluster i'm trying to use only has python 2.6.6
# consequently, cannot get new version of scipy with skewnorm distribution
# so reimplementing certain features of this distribution
import numpy as np
from scipy.stats import norm

class skewnorm:
    def __init__(self):
        """Nothing to really init here..."""
        
    def pdf(self, x, loc=0.0, a=0.0, scale=1.0):
        return (2*norm.pdf((x - loc)/scale)*norm.cdf(a * (x-loc)/scale) 
            / scale)
        
    def rvs(self, size, a=0.0, loc=0.0, scale=1.0):
#        from scipy source..
#        u0 = self._random_state.normal(size=self._size)
#        v = self._random_state.normal(size=self._size)
#        d = a/np.sqrt(1 + a**2)
#        u1 = d*u0 + v*np.sqrt(1 - d**2)
#        return np.where(u0 >= 0, u1, -u1)
# also, algorithm described here: http://azzalini.stat.unipd.it/SN/faq-r.html
        u0 = np.random.normal(scale=scale, size=size)
        v = np.random.normal(scale=scale, size=size)
        d = a/np.sqrt(1+(a)**2)
        u1 = d*u0 + v*np.sqrt(1-d**2)
        return np.where(u0 >= 0, u1, -u1) + loc