import numpy as np
from classes import *
from filters import *
from GA_functions import *
from plotting_car import *
from Problems.car_simple import *
from helpers import seedRNG

'''
Holonomic Robot Simple Example

This is an extremely simple linear system where we have effectively a point robot with 
pose (x, y, theta), and because it is holonomic, it can move independently in each of the
three DOFs. Therefore, there are no nonlinear sinusoids that need to be linearized with
an EKF

We assume that the state is directly observable, but with a considerable amount of noise. 
So, this is like we have a noisy GPS estimate of the x,y position with a compass telling
us our heading angle

'''

# Defining the genetic algorithm selection, crossover, and mutation methods
selection_fxn = rouletteWheelSelection
crossover_fxn = interpolationCrossover
mutation_fxn = gaussianMutation

# Building the classes and running the simulation/filter/GA

seedRNG(0) # Seed the random number gen. for repeatable testing

# Sample the priors
x0 = np.random.multivariate_normal(mu0_state, sigma0_state) # Sample x0 from the prior

# Build our models
time_model = TimeModel(dt, duration=20)
dynamics_model = LinearDynamicsModel(A, B)
measurement_model = LinearMeasurementModel(C)
controls_model = ControlsModel(uHistory=np.vstack((vx_avg*np.sin(time_model.times), 
                                                   vy_avg*np.cos(time_model.times), 
                                                   phi_avg*np.sin(time_model.times))))
noise_model = NoiseModel(Q, R)
P = Problem(dynamics_model, measurement_model, controls_model, noise_model, time_model)
sim = SimulatedResult(x0, P)

GAinit = GAInitialization(mu0_state, sigma0_state, mu0_chromosome, sigma0_chromosome, 
                          time_model.nTimesteps, pop_size, selection_fxn, crossover_fxn, mutation_fxn, k_max, 
                          k_selection=None, L_interp_crossover=0.5, mutation_stdev=mutation_stdev)
GA = GAResult(KF, P, sim.yHistory, GAinit)

# Plotting
last_gen_best_ind = np.argmin(GA.evalHistory[-1])
muHistory_best = GA.muHistories[np.argmin(GA.evalHistory[-1])]
plot_best_filter_vs_truth(time_model.times, sim.xHistory, GA.muHistories[last_gen_best_ind], GA.sigmaHistories[last_gen_best_ind])
plot_generation(sim.xHistory, GA.muHistories_history, GA.evalHistory, gen=0) # Try adjusting the generation
plot_mahalanobis_convergence(GA.evalHistory)
plot_chromosome_convergence(GA.bestHistory, true_chromosome)