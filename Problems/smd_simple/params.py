import numpy as np

# For simple spring mass damper system

dt = 0.1
k = 5
c = 1
m = 10
natural_freq = np.sqrt(k/m)
forcing_freq = 1.5*natural_freq

xdim = 2
ydim = 1
Q = 0.1 * np.eye(xdim) * dt
R = 0.1 * np.eye(ydim) 


mu0 = np.array([0,0])
sigma0 = 0.01 * np.eye(xdim)

# Other constants / useful values
duration = 50 # Length of the simulation
nTimesteps = np.floor(duration / dt).astype('int32')

# Do we need these??
eps = 1e-3 # For iEKF
maxIterations = 30 # For iEKF