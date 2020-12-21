# Rylan Andrews
# 10-13-2020
# Classical Solution to the 2-body problem using numerical integration

# Imports
import scipy as sci
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Universal Gravitational Constant
G = 6.67408e-11     #N-m2/kg2

# Mass of sun
m_nd = 1.989e+30                            # kg
# Distance between stars in Alpha Centauri
r_nd = 5.326e+12                            # m
# Relative velocity of earth around the sun
v_nd = 30_000                               # m/s
# Orbital period of Alpha Centauri
t_nd = 79.91 * 365 * 24 * 3_600 * 0.51      # s

K1 = ( G * t_nd * m_nd ) / ( r_nd**2 * v_nd )
K2 = ( v_nd * t_nd ) / r_nd

# Mass of Alpha Centauri A
m1 = 1.0
# Mass of Alpha Centauri B
m2 = 1.0

# Initial positions
r1 = [0.5, 0]
r2 = [-0.5, 0]

# Convert position vectors to arrays
r1 = np.array(r1, dtype="float64")
r2 = np.array(r2, dtype="float64")

# Find center of mass
r_com = ( (m1 * r1) + (m2 * r2))

# Define intial velocities
v1 = [0, 0.01]
v2 = [0, -0.01]

# Convert velocity vectors to arrays
v1 = np.array(v1, dtype="float64")
v2 = np.array(v2, dtype="float64")

# Find velocities of center of mass
v_com = ( ( m1 * v1 ) + ( m2 * v2 ) ) / ( m1 + m2 )

# Model function for odeint solver defining equations of motion
def TwoBodyEquations(w, t, G, m1, m2):
    # Unpack values from odeint solver
    r1 = w[:2]
    r2 = w[2:4]
    v1 = w[4:6]
    v2 = w[6:8]

    # Calculate magnitude or norm of vector
    r = sci.linalg.norm(r2-r1)

    # Differential equations
    dv1dt = (K1 * m2 * (r2-r1)) / r**3
    dv2dt = (K1 * m1 * (r1-r2)) / r**3
    dr1dt = K2 * v1
    dr2dt = K2 * v2

    r_derivs = np.concatenate((dr1dt, dr2dt))
    derivs = np.concatenate((r_derivs, dv1dt, dv2dt))
    return derivs

# Package initial parameters
init_params = np.array([r1, r2, v1, v2])
init_params = init_params.flatten()

time_span = np.linspace(0, 8, 5000)

# Solve problem
two_body_sol = odeint(TwoBodyEquations, init_params, time_span, args=(G, m1, m2))

# Unpack results
r1_sol_x = two_body_sol[:,0]
r1_sol_y = two_body_sol[:,1]
r2_sol_x = two_body_sol[:,2]
r2_sol_y = two_body_sol[:,3]

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
line1, = ax.plot(r1[0], r1[1])
line2, = ax.plot(r2[0], r2[1])

def animation_frame(i):
    line1.set_data(r1_sol_x[:i], r1_sol_y[:i])
    line2.set_data(r2_sol_x[:i], r2_sol_y[:i])
    return line1,line2

animation = FuncAnimation(fig, func=animation_frame, frames=range(len(r1_sol_x)), interval=10)


plt.show()