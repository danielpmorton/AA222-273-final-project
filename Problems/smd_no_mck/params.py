import numpy as np

# Spring-Mass-Damper with unknown mass, damping coeff, and spring constant

dt = 0.1
k_unknown = 5
c_unknown = 1
m_unknown = 10
natural_freq = np.sqrt(k_unknown/m_unknown)
forcing_freq = 1.5*natural_freq

xdim = 5 # Since we have position, velocity, and m, c, k in the state
ydim = 1 # Still just recording the position of the mass only
Q = 0.1 * np.eye(xdim) * dt
R = 0.1 * np.eye(ydim) 

# When simulating, there is no noise added to the unknown mass
Q_sim = 0.1 * dt * np.diag([1, 1, 0, 0, 0])
sigma0_sim = 0.01 * np.diag([1, 1, 0, 0, 0])
mu0_sim = np.array([0, 0, m_unknown, c_unknown, k_unknown])

mu0 = np.array([0, 0, m_unknown, c_unknown, k_unknown]) # Try initializing these off of their real values
sigma0 = 0.01 * np.eye(xdim)
# mu0 = np.array([0, 0, 1.5*m_unknown, 1.5*c_unknown, 1.5*k_unknown])
# sigma0 = np.array([0.01, 0.01, 5**2, .5**2, 2.5**2])

# Other constants / useful values
duration = 50 # Length of the simulation
nTimesteps = np.floor(duration / dt).astype('int32')

# Do we need these??
eps = 1e-3 # For iEKF
maxIterations = 30 # For iEKF