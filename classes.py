import numpy as np
import scipy.linalg

class SigmaPoints:
    def __init__(self, mu, sigma, L=2):
        self.mu = mu
        self.sigma = sigma
        self.L = L # lambda
        self.points, self.weights = self.getPointsAndWeights()
        
    def getPointsAndWeights(self):
        xdim = len(self.mu)
        points = [] # A list of np arrays
        weights = [] # A list of values
        # Handle the central point
        center = self.mu
        centerweight = self.L / (self.L + xdim)
        points.append(center)
        weights.append(centerweight)
        # Handle the other points around the center
        for i in range(xdim):
            offset = scipy.linalg.sqrtm((self.L + xdim)*self.sigma)[:,i]
            point1 = self.mu + offset
            point2 = self.mu - offset
            weight = 1/(2 * (self.L + xdim))
            points.extend((point1, point2))
            weights.extend((weight, weight))
        return points, weights

class ParticleSet:
    def __init__(self, mu0, sigma0, numParticles):
        self.particles = np.random.multivariate_normal(mu0, sigma0, size=numParticles).T # With the transpose, shape = (xdim,numParticles)
        self.weights = 1/numParticles * np.ones(numParticles) # size (numParticles,)

# Combines the dynamics, measurement, controls, noise, and time into a single Problem model
class Problem:
    def __init__(self, dynamics_model, measurement_model, controls_model, noise_model, time_model):
        self.dynamics_model = dynamics_model
        self.measurement_model = measurement_model
        self.controls_model = controls_model
        self.noise_model = noise_model
        self.time_model = time_model

class MeasurementModel:
    def __init__(self, g, C, ydim):
        self.g = g 
        self.C = C 
        self.ydim = ydim

    def g(self, x):
        return self.g(x)

    def C(self, x):
        return self.C(x)

class DynamicsModel:
    def __init__(self, f, A, xdim):
        self.f = f # Callable. f(x, u)
        self.A = A # Callable. A(x, u)
        self.xdim = xdim
    
    def f(self, x, u):
        return self.f(x, u)
    
    def A(self, x, u):
        return self.A(x, u)

class NoiseModel:
    def __init__(self, Q, R):
        self.Q = Q
        self.R = R

class ControlsModel:
    # Can initialize this with either the full uHistory array
    # Or, a function applied over an array of times
    def __init__(self, uHistory=None, fxn=None, times=None):
        self.fxn = fxn
        self.times = times

        if uHistory is not None:
            self.uHistory = uHistory
        elif fxn is not None and times is not None:
            dim = 1 if np.isscalar(fxn(times[0])) else len(fxn(times[0]))
            self.uHistory = np.zeros((dim, len(times)))
            for i,t in enumerate(times):
                self.uHistory[:,i] = self.fxn(t)
        else:
            raise Exception("Must provide more inputs to the controls model")

class TimeModel:
    # Must provide dt and one of the three optional inputs
    def __init__(self, dt, nTimesteps=None, times=None, duration=None):
        self.dt = dt
        if duration != None:
            self.duration = duration
            self.nTimesteps = np.floor(self.duration / self.dt).astype('int32')
            self.times = self.dt * np.arange(self.nTimesteps)
        elif nTimesteps != None:
            self.nTimesteps = nTimesteps
            self.duration = self.dt * self.nTimesteps
            self.times = self.dt * np.arange(self.nTimesteps)
        elif times != None:
            self.times = times
            self.nTimesteps = len(times)
            self.duration = self.dt * self.nTimesteps
        else:
            raise Exception("Must provide more info to the time model!")


class FilterStorage:
    def __init__(self, mu0, sigma0, nTimesteps):
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.nTimesteps = nTimesteps
        self.muHistory, self.sigmaHistory = self.initializeStorage(mu0, sigma0, nTimesteps)
    
    def initializeStorage(self, mu0, sigma0, nTimesteps):
        dim = len(mu0)
        muHistory = np.zeros((dim, nTimesteps))
        sigmaHistory = np.zeros((dim, dim, nTimesteps))
        muHistory[:,0] = mu0
        sigmaHistory[:,:,0] = sigma0
        return muHistory, sigmaHistory

class PFStorage(FilterStorage):
    def __init__(self, mu0, sigma0, nTimesteps, numParticles):
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.numParticles = numParticles
        self.X = self.initParticleSet(mu0, sigma0, numParticles)
        self.muHistory, self.sigmaHistory = self.initializeStorage(mu0, sigma0, nTimesteps)

    def initParticleSet(mu0, sigma0, numParticles):
        particles = np.random.multivariate_normal(mu0, sigma0, size=numParticles).T # With the transpose, shape = (state dim,1000)
        weights = 1/numParticles * np.ones(numParticles)
        X = ParticleSet(particles, weights)
        return X

class SimulatedResult:
    def __init__(self, x0, problem):
        self.x0 = x0
        self.uHistory = problem.controls_model.uHistory
        self.dt = problem.time_model.dt
        self.nTimesteps = problem.time_model.nTimesteps
        self.times = problem.time_model.times
        self.R = problem.noise_model.R
        self.Q = problem.noise_model.Q
        self.xdim = len(self.x0)
        self.ydim = self.R.shape[0]
        self.f = problem.dynamics_model.f
        self.g = problem.measurement_model.g
        self.xHistory, self.yHistory = self.simulate()

    def simulate(self):
        xHistory = np.zeros((self.xdim, self.nTimesteps))
        xHistory[:,0] = self.x0
        yHistory = np.zeros((self.ydim, self.nTimesteps))

        for i in range(1, self.nTimesteps):
            # Retrieve
            xprev = xHistory[:,i-1]
            u = self.uHistory[:,i]
            # Calculate noise
            v = np.random.multivariate_normal(np.zeros(self.ydim), self.R)
            w = np.random.multivariate_normal(np.zeros(self.xdim), self.Q)
            # Calculate state and measurement
            xnew = self.f(xprev, u) + w
            y = self.g(xnew) + v
            # Store data
            xHistory[:,i] = xnew
            yHistory[:,i] = y
        
        print("Simulation complete")
        return xHistory, yHistory

class FilterResult:
    def __init__(self, filter, problem, measurement_history, filter_storage, UKF_lambda=2, iEKF_maxIter=None, iEKF_eps=None, PF_numParticles=None, PF_resample=True):
        self.Q = problem.noise_model.Q
        self.R = problem.noise_model.R
        self.f = problem.dynamics_model.f
        self.A = problem.dynamics_model.A
        self.g = problem.measurement_model.g
        self.C = problem.measurement_model.C
        self.uHistory = problem.controls_model.uHistory
        self.yHistory = measurement_history
        self.filter = filter
        self.muHistory = filter_storage.muHistory # Initial value
        self.sigmaHistory = filter_storage.sigmaHistory # Initial value
        self.nTimesteps = problem.time_model.nTimesteps
        self.UKF_lambda = UKF_lambda # Default value of 2 since this is standard
        self.iEKF_maxIter = iEKF_maxIter
        self.iEKF_eps = iEKF_eps
        self.PF_numParticles = PF_numParticles
        self.PF_resample = PF_resample
        self.runFilter()

    def runFilter(self):
        # Initialize mu and sigma to the mu0, sigma0 values stored
        mu = self.muHistory[:,0]
        sigma = self.sigmaHistory[:,:,0]
        # If we have a PF, need to also initialize a particle set
        if self.filter.__name__.lower() == "pf":
            X = ParticleSet(self.mu, self.sigma, self.PF_numParticles)
        # Filter for all timesteps
        for i in range(1, self.nTimesteps):
            # Retrieve from simulation data
            y = self.yHistory[:,i]
            u = self.uHistory[:,i]
            # Filter - TODO: need to check if the filter name thing works
            if self.filter.__name__.lower() == "ekf":
                mu, sigma = self.filter(mu, sigma, u, y, self.Q, self.R, self.f, self.g, self.A, self.C)
            elif self.filter.__name__.lower() == "ukf":
                mu, sigma = self.filter(mu, sigma, u, y, self.Q, self.R, self.f, self.g, self.UKF_lambda)
            elif self.filter.__name__.lower() == "iekf":
                mu, sigma = self.filter(mu, sigma, u, y, self.Q, self.R, self.f, self.g, self.A, self.C, self.iEKF_maxIter, self.iEKF_eps)
            elif self.filter.__name__.lower() == "pf":
                # Note that f and g need to be specific to PF!
                mu, sigma = self.filter(X, u, y, self.Q, self.R, self.f, self.g, self.PF_numParticles, self.PF_resample)
            else:
                raise Exception("Invalid filter name")
            # Store
            self.muHistory[:,i] = mu
            self.sigmaHistory[:,:,i] = sigma
        
        print("Filtering complete")

