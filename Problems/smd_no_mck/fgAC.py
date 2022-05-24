import numpy as np
from Problems.smd_no_mck.params import *

def f(x, u):
    pos, vel, m, c, k = x
    force = u[0]
    d_pos = vel*dt
    d_vel = (1/m) * (force - c*vel - k*pos) * dt
    d_mass = 0
    d_c = 0
    d_k = 0
    dx = np.array([d_pos, d_vel, d_mass, d_c, d_k])
    return x + dx

def g(x):
    pos, vel, m, c, k = x
    return pos

def A(x, u):
    pos, vel, m, c, k = x
    force = u[0]
    return np.array([[      1,       dt,                                     0,         0,         0], 
                     [-k*dt/m, 1-c*dt/m, (-1/(m**2)) * (force - c*vel - k*pos), -vel*dt/m, -pos*dt/m], 
                     [      0,        0,                                     1,         0,         0], 
                     [      0,        0,                                     0,         1,         0], 
                     [      0,        0,                                     0,         0,         1]])

def C(x):
    return np.array([[1, 0, 0, 0, 0]])