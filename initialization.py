#
# initialization.py
#
# created 20-07-20 gcr
#
# handles shared setup stuff that would otherwise appear in a bunch of different places
# by having a unified, shared initialization process, protect against version skew
#
import numpy as np
from constants.constants import (masses, distances, physics, tofWindows)
from constants.constants import experimentConsts

class initialize_oneBD:

    @staticmethod
    def setupDeuteronBinning(nBins = 400, eD_minRange = 200., eD_maxRange = 2200.):
        # range of eD expanded for seemingly higher energy oneBD neutron beam

        eD_range = (eD_minRange, eD_maxRange)
        eD_binSize = (eD_maxRange - eD_minRange)/nBins
        eD_binCenters = np.linspace(eD_minRange + eD_binSize/2,
                                    eD_maxRange - eD_binSize/2,
                                    nBins)
        return nBins, eD_range, eD_binSize, eD_binCenters


    @staticmethod
    def setupXbinning(nBins = 20, x_minRange = 0., x_maxRange = distances.tunlSSA_CsI_oneBD.cellLength):

   
        x_range = (x_minRange,x_maxRange)
        x_binSize = (x_maxRange - x_minRange)/nBins
        x_binCenters = np.linspace(x_minRange + x_binSize/2,
                                x_maxRange - x_binSize/2,
                                nBins)
        return nBins, x_range, x_binSize, x_binCenters

    @staticmethod
    def getCellAttenuationCoeffs( xpoints ):
        """
        Returns array of weighting factors introducing attenuation along length of gas cell.
        """
        return np.exp(-xpoints / experimentConsts.csi_oneBD.gasCellAttentuationLength)