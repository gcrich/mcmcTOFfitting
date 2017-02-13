# constants.py
#
# created dec 2016 through 'this is how to do python right' contributions
# of m.d. medley
#
# values etc from g.c. rich
import scipy.constants as scipyConsts


class physics(object):
    """Constants related to physics
    """
    speedOfLight = 29.9792  # in cm/ns
    epsilon_0 = scipyConsts.epsilon_0 * 1e-2 # in F/cm
    molarMassConstant = 1 # in g/mol


class masses(object):
    """Constants related to masses
    """
    electron = 511 # keV/c^2
    deuteron = 1.8756e+06  # keV /c^2
    neutron = 939565.0  # keV/c^2
    he3 = 2.809414e6  # keV/c^2


class distances(object):
    """Distances related to a given experiment
    """
    class tunlSSA_CsI(object):
        """Distances associated with the Jan 2016 CsI QF run at TUNL SSA
        """
        # cm, distance from tipof gas cell to 0deg face
        cellToZero = 518.055
        cellLength = 2.86  # cm, length of gas cell
        zeroDegLength = 3.81  # cm, length of 0deg detector
        tipToColli = 148.4 # cm, distance from cell tip to collimator exit
        colliToZero = 233.8 # cm, distance from collimator exit to face of
            # 0deg AT ITS CLOSEST LOCATION
        delta1 = 131.09 # cm, difference between close and mid 0degree loc.
        delta2 = 52.39 # cm, difference between mid and far 0deg loc

        standoffClose = tipToColli + colliToZero
        standoffMid = standoffClose + delta1
        standoffFar = standoffMid + delta2
        
        colliToCsI = 59.45 #from collimator face to closest side of CsI
        csiToZero = 355.7 # from closest side of CsI to face of zero degree
        csiDiameter = 2.341 # cm, measured by calipers sep8 2014 (picture)
        standoff_TUNLruns = colliToCsI + csiToZero + csiDiameter + tipToColli


class qValues(object):
    """Q values of reactions
    """
    ddn = 3268.914  # Q value of DDN reaction, in keV

    
class tofWindows(object):
    """Windows of TOF used for different standoff distances
    """
    nBins = {'close': 45, 'mid': 50, 'far': 70, 'production': 65}
    maxRange = {'close': 175.0, 'mid': 225.0, 'far': 260.0, 'production': 260.0}
    minRange = {'close':130.0, 'mid': 175.0, 'far': 190.0, 'production':195.0}