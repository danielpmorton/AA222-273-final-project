import numpy as np
import scipy.linalg

# Variables/constants specified in the problem statement
xdim = 11
posedim = 3
mapdim = 8
ydim = 12
dt = 0.1
Q = 0.1 * np.eye(xdim) * dt
R = 0.1 * np.eye(ydim) 
vt = 1
mu0 = np.concatenate((np.zeros(posedim), np.array([0,0,10,0,10,10,0,10]))) # Merged mu0 for pose and map
sigma0 = 0.01 * np.eye(xdim)
# Other constants / useful values
duration = 20 # Length of the simulation
nTimesteps = np.floor(duration / dt).astype('int32')
numMarkers = 4
eps = 1e-3 # For iEKF
maxIterations = 30 # For iEKF

verySmallValue = 1e-6 # For division stability

# SIMULATION-ONLY STUFF
# This assumes that we know the map exactly
Q_sim = scipy.linalg.block_diag(0.1*dt*np.eye(posedim), np.zeros((mapdim, mapdim)))
sigma0_sim = scipy.linalg.block_diag(0.01*np.eye(posedim), np.zeros((mapdim, mapdim)))
markerLocs = [np.array([0,0]), np.array([10, 0]), np.array([10,10]), np.array([0,10])]

# NOTE:
# x0 and x0_sim have been moved to the main file due to havign the randomness evaluated only once