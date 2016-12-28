#
# ionStopping.py
#
# created winter 2016, gcrich
#
# has classes/functions for handling ion stopping in materials
# ie implementation of bethe function ODE
#
# TODO: implement corrections in bethe stopping
# TODO: check that certain fixed values are appropriate to have fixed
# TODO: convenience functions to accept simpler args when initializing models
#       e.g., specify 'D2,0.5atm', and look up appropriate parameters
#       this is a 'maybe'

from constants.constants import (physics,masses)
import numpy as np
import scipy.constants as sciConsts

# something worth noting, mean excitation potential for D2 is 19.2
# according to: http://pdg.lbl.gov/2016/AtomicNuclearProperties/HTML/deuterium_gas.html
class ionStopping:
    """
    Handles ion stopping in materials
    """
    def dEdx(self, energy, x=0):
        """
        Calculate the stopping power at a given energy (in keV/cm)
        This function needs to be implemented by each stopping model
        """
        return 0.0
    
    class simpleBethe:
        """
        Simple implementation of Bethe formula
        """
        def __init__(self, params):
            """
            simpleBethe class needs parameters: proton number, mass number, density, charge of incident ion, and mean excitation energy
            """
            Z, A, rho, charge, excitation = params
            self.protonNumber = Z
            self.massNumber = A
            self.density = rho
            self.ionCharge = charge
            self.iExcitation = excitation
            # go ahead and calculate number density of the stopping medium
            # it won't change and will be used for every dEdx calc
            self.numDensity = (sciConsts.Avogadro * self.protonNumber *
                              self.density /
                              (self.massNumber*physics.molarMassConstant))
                              
            # this factor is (e**2/(4 pi epsilon_0))**2
            # it doesn't depend on anything user will enter
            # depending on unit system, perhaps a different value is needed
            # but we're going all keV-cm-ns
            # so use PEP8 naming convention for consts, which is ugly all caps
            self.FIXED_FACTOR = 1.67489e-14 # worked out on paper
        
        
        def dEdx(self, energy, x=0):
            """Calculate the stopping power at a given energy (in keV/cm)"""
#            print('params:\nproton number {}\nA {}\nrho {}\ncharge {}\nI {}\nn Density {}'.format(self.protonNumber, self.massNumber, self.density, self.ionCharge, self.iExcitation, self.numDensity))
            velocity = (np.sqrt(2 * energy / masses.deuteron) *
                        physics.speedOfLight)
#            print('velocity for energy {} is {}'.format(energy,velocity))
            leadingTerm = ( 4 * np.pi * self.numDensity * self.ionCharge**2 /
                           (masses.electron * physics.speedOfLight**2 *
                           velocity**2))
            logArg = (2 * masses.electron /(physics.speedOfLight**2)*
                      velocity**2 / (self.iExcitation*1e-3))
            stopping = -1 * leadingTerm * self.FIXED_FACTOR * np.log(logArg)
#            print('leading term {}\nfixed {}\nlog {}'.format(leadingTerm,
#                                                             self.fixedFactor,
#                                                             np.log(logArg)))
            return stopping