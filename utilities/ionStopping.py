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
from scipy.integrate import ode
from scipy.interpolate import RectBivariateSpline

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
            self.materials = []
            if len(params) == 5:
                Z, A, rho, charge, excitation = params
                # go ahead and calculate number density of the stopping medium
                # it won't change and will be used for every dEdx calc
                
                # this was originally here
                # the "self" references don't.. seem right 
                # left here in case i'm missing something
                #eNumDensity = (sciConsts.Avogadro * self.protonNumber *
                #              self.density /
                #              (self.massNumber*physics.molarMassConstant))
                eNumDensity = (sciConsts.Avogadro * Z *
                              rho /
                              (A * physics.molarMassConstant))
            
                self.materials.append([Z, A, rho, eNumDensity, excitation])
            
            else:
                charge = params[0]
            self.ionCharge = charge
            
            # this factor is (e**2/(4 pi epsilon_0))**2
            # it doesn't depend on anything user will enter
            # depending on unit system, perhaps a different value is needed
            # but we're going all keV-cm-ns
            # so use PEP8 naming convention for consts, which is ugly all caps
            self.FIXED_FACTOR = 1.67489e-14 # worked out on paper
        
        def addMaterial(self, material):
            matZ, matA, matRho, excitation = material
            eNumDensity = (sciConsts.Avogadro * matZ *
                              matRho /
                              (matA*physics.molarMassConstant))
            self.materials.append([matZ, matA, matRho, eNumDensity, excitation])
        
        def dEdx(self, energy, x=0):
            """Calculate the stopping power at a given energy (in keV/cm)"""
#            print('params:\nproton number {}\nA {}\nrho {}\ncharge {}\nI {}\nn Density {}'.format(self.protonNumber, self.massNumber, self.density, self.ionCharge, self.iExcitation, self.numDensity))
            
            velocity = (np.sqrt(2 * energy / masses.deuteron) *
                                physics.speedOfLight)
            leadingTerm = ( 4 * np.pi * self.ionCharge**2 /
                           (masses.electron * physics.speedOfLight**2 *
                           velocity**2))
            fractionalContributions = 0
            for component in self.materials:
                eNumDensity, avgExcitation = component[-2:]
                logArg = (2 * masses.electron /(physics.speedOfLight**2)*
                      velocity**2 / (avgExcitation))
                fractionalContributions= fractionalContributions + (eNumDensity * np.log(logArg))
            stopping = -1 * leadingTerm * self.FIXED_FACTOR * fractionalContributions
#            print('leading term {}\nfixed {}\nlog {}'.format(leadingTerm,
#                                                             self.fixedFactor,
#                                                             np.log(logArg)))
            return stopping

            


    class betheApprox:
        """
        Approximate Bethe stopping in specific case, using a bivariate spline
        """
        

        def __init__(self, modelToApproximate, eD_binInfo, xBinCenters):
            self.betheModel = modelToApproximate
            self.x_binCenters = xBinCenters
            dedxfxn = self.betheModel.dEdx
            dedxForODE = lambda x, y: dedxfxn(energy=y,x=x)

            eD_minRange, eD_maxRange, eDstep = eD_binInfo
            #eDstep = 50
            #eD_minRange = 400.
            #eD_maxRange = 1200.
            self.edgrid = np.arange(eD_minRange, eD_maxRange, eDstep)
            
            print(self.edgrid)

            z = np.zeros(len(self.x_binCenters))

            for eZero in self.edgrid:
                solver = ode(dedxForODE).set_integrator('dopri5').set_initial_value(eZero)
                thisSolution = np.array([solver.integrate(x) for x in self.x_binCenters])
                z = np.vstack((z,thisSolution.flatten()))
            self.z = z[1:]

            self.stoppingSpline = RectBivariateSpline( self.edgrid, self.x_binCenters, self.z)

        def evalStopped(self, eZero, xLoc):
            """
            Evaluate the stopped beam at different locations
            """
            return self.stoppingSpline(eZero, xLoc)

def getHavarStopping():
    materials = []
    # Linear Formula: Co 42.5% / Cr 19.5% / Ni 12.7% / W 2.8 % / Mo 2.6% / Mn 1.6% / Fe- balance (18.3%)
    atomicMasses= []
    atomicMasses.append([27, 58.933195])
    atomicMasses.append([24, 51.9961])
    atomicMasses.append([28, 58.6934])
    atomicMasses.append([74, 183.84])
    atomicMasses.append([42, 95.94])
    atomicMasses.append([25, 54.938045])
    atomicMasses.append([26, 55.845])
    atomicMasses.append([6, 12.011])
    atomicFractions = {}
# these are based on american elements website
#    atomicFractions.update({27: 0.425})
#    atomicFractions.update({24: 0.195})
#    atomicFractions.update({28: 0.127})
#    atomicFractions.update({74: 0.028})
#    atomicFractions.update({42: 0.026})
#    atomicFractions.update({25: 0.016})
#    atomicFractions.update({26: 0.183})
# these are based on SRIM
    atomicFractions.update({27: 0.417829})
    atomicFractions.update({24: 0.222858})
    atomicFractions.update({28: 0.128336})
    atomicFractions.update({74: 0.008824})
    atomicFractions.update({42: 0.014494})
    atomicFractions.update({25: 0.016874})
    atomicFractions.update({26: 0.181139})
    atomicFractions.update({6: 0.009648})
    excitationEnergies = {}
    excitationEnergies.update({27: 0.2970})
    excitationEnergies.update({24: 0.2570})
    excitationEnergies.update({28: 0.3110})
    excitationEnergies.update({74: 0.7270})
    excitationEnergies.update({42: 0.4240})
    excitationEnergies.update({25: 0.2720})
    excitationEnergies.update({26: 0.2860})
    excitationEnergies.update({6: 0.078})
    #havar = absorberMaterial(absorberDensity = 8.3)
    havar = ionStopping.simpleBethe([1])
    for mat in atomicMasses:
        z = mat[0]
        materials.append([z, mat[1], 8.3 * atomicFractions[z], excitationEnergies[z]])
    for mat in materials:
        havar.addMaterial( mat )
    return havar