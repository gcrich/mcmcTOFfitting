# Example constants file with constants in namespaces


class Physics(object):
    """
    Constants related to physics
    """
    speedOfLight = 29.9792  # in cm/ns


class Masses(object):
    """
    Constants related to masses
    """
    mass_deuteron = 1.8756e+06  # keV /c^2
    mass_neutron = 939565.0  # keV/c^2
    mass_he3 = 2.809414e6  # keV/c^2


class Distances(object):
    """
    Constants related to distances
    """
    # cm, distance from tipof gas cell to 0deg face
    distance_cellToZero = 518.055
    distance_cellLength = 2.86  # cm, length of gas cell
    distance_zeroDegLength = 3.81  # cm, length of 0deg detector


class Energy(object):
    """
    Constants related to energy
    """
    qValue_ddn = 3268.914  # Q value of DDN reaction, in keV
