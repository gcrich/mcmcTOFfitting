# They are imported into other modules like so...
from constants.tof_constants import (Distances, Energy, Physics, Masses)

# Then you access them within their namespaces like so...
print 'Cell length is: {0}'.format(Distances.distance_cellLength)
print 'The Q value of a DDN reaction is: {0}'.format(Energy.qValue_ddn)
print 'The speed of light is: {0}'.format(Physics.speedOfLight)
print 'The mass of Deuteron is: {0}'.format(Masses.mass_deuteron)
