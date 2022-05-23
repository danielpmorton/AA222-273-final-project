import numpy as np
from classes import *

# 1D Spring mass damper system
# State: [position x, velocity v]
# Measurement: position only

# Parameters:
dt = 0.1
k = 5
c = 1
m = 10
natural_freq = np.sqrt(k/m)
forcing_freq = 1.5*natural_freq

def force(t):
    return np.sin(forcing_freq * t)

def f(x,u):
    pos, vel = x
    force = u
    d_pos = vel*dt
    d_vel = (1/m) * (force - c*vel - k*x) * dt
    dx = np.array([d_pos, d_vel])
    return x + dx

def A(x, u):
    pos, vel = x
    return np.array([[      1,       dt], 
                     [-k*dt/m, 1-c*dt/m]])

def g(x):
    pos, vel = x
    return pos

def C(x):
    return np.array([1, 0])