import numpy as np
from Problems.smd_no_m.params import *

def f(x, u):
    pos, vel, m = x
    force = u[0]
    d_pos = vel*dt
    d_vel = (1/m) * (force - c*vel - k*pos) * dt
    d_mass = 0
    dx = np.array([d_pos, d_vel, d_mass])
    return x + dx

def g(x):
    pos, vel, m = x
    return pos

def A(x, u):
    pos, vel, m = x
    force = u[0]
    return np.array([[      1,       dt,                                     0], 
                     [-k*dt/m, 1-c*dt/m, (-1/(m**2)) * (force - c*vel - k*pos)], 
                     [      0,        0,                                     1]])

def C(x):
    return np.array([[1, 0, 0]])