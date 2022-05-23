import numpy as np

# Spring-Mass-Damper with unknown mass

dt = 0.1
k = 5
c = 1
m_unknown = 10
natural_freq = np.sqrt(k/m_unknown)
forcing_freq = 1.5*natural_freq

xdim = 3 # Since we have position, velocity, and mass in the state now
ydim = 1 # Still just recording the position of the mass only
Q = 0.1 * np.eye(xdim) * dt
R = 0.1 * np.eye(ydim) 

# When simulating, there is no noise added to the unknown mass
Q_sim = 0.1 * dt * np.diag([1, 1, 0])
sigma0_sim = 0.01 * np.diag([1, 1, 0])
mu0_sim = np.array([0, 0, 10])

mu0 = np.array([0, 0, 10]) # Check trying initializing m past m
sigma0 = 0.01 * np.eye(xdim)

# Other constants / useful values
duration = 50 # Length of the simulation
nTimesteps = np.floor(duration / dt).astype('int32')

# Do we need these??
eps = 1e-3 # For iEKF
maxIterations = 30 # For iEKF