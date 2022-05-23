import numpy as np
from Problems.smd_simple.params import *

def f(x,u):
    pos, vel = x
    force = u[0]
    d_pos = vel*dt
    d_vel = (1/m) * (force - c*vel - k*pos) * dt
    dx = np.array([d_pos, d_vel])
    return x + dx

def g(x):
    pos, vel = x
    return pos

def A(x, u):
    pos, vel = x
    return np.array([[      1,       dt], 
                     [-k*dt/m, 1-c*dt/m]])

def C(x):
    return np.array([[1, 0]])