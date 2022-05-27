



selection_fxn = rouletteWheelSelection
crossover_fxn = interpolationCrossover
mutation_fxn = gaussianMutation
pop_size = 20 # Number of KFs we have running
k_max = 10 # Maximum number of iterations in the GA
batch_size = 5 # Number of KF to sample to run the GA
num_T = 10 # Number of timesteps to propagate before running GA
mean_A = np.array([1,2,3,4]) # Remember to update this
cov_A = np.eye(4) # Remember to update this