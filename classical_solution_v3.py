# Rylan Andrews
# 10/14/2020
# Three body problem solver engine

# Math imports
import scipy as sci
import numpy as np
from scipy.integrate import odeint

# Visualization imports
import matplotlib.pyplot as plt 
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation

# Performance imports 
import time

# Writing to csv
import csv

class ThreeBodyProblem:
    """An engine for solving 3 body problems"""

    # Constants
    # Universal Gravitational Constant
    G = 6.67408e-11                             #N-m2/kg2
    # Mass of sun
    m_nd = 1.989e+30                            # kg
    # Distance between stars in Alpha Centauri
    r_nd = 5.326e+12                            # m
    # Relative velocity of earth around the sun
    v_nd = 30_000                               # m/s
    # Orbital period of Alpha Centauri
    t_nd = 79.91 * 365 * 24 * 3_600 * 0.51      # s

    # Simplify constants into one variable for easy use in equations
    K1 = ( G * t_nd * m_nd ) / ( r_nd**2 * v_nd )
    K2 = ( v_nd * t_nd ) / r_nd


    def __init__(self, orbital_periods, m1, m2, m3, p1, p2, p3, v1, v2, v3):
        """Inititalize ThreeBodyProblem object"""
        self.orbital_periods = orbital_periods
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

        # Convert position vectors to arrays
        p1 = np.array(p1, dtype="float64")
        p2 = np.array(p2, dtype="float64")
        p3 = np.array(p3, dtype="float64")

        # Find Center of Mass
        self.p_com = ( (m1 * p1) + (m2 * p2) + (m3 * p3) ) / (m1 + m2 + m3)

        # Convert velocity vectors to arrays
        v1 = np.array(v1, dtype="float64")
        v2 = np.array(v2, dtype="float64")
        v3 = np.array(v3, dtype="float64")

        # Find Velocity of Center of Mass
        self.v_com = ( ( m1 * v1 ) + ( m2 * v2 ) + ( m3 * v3 )) / ( m1 + m2 + m3 )


    def equations(self, w, t, G, m1, m2, m3):
        """Equation model to be passed to odeint solver"""
        # Unpack data
        p1 = w[:2]
        p2 = w[2:4]
        p3 = w[4:6]
        v1 = w[6:8]
        v2 = w[8:10]
        v3 = w[10:12]

        # Find distance between bodies
        p12 = sci.linalg.norm(p2-p1)
        p13 = sci.linalg.norm(p3-p1)
        p23 = sci.linalg.norm(p3-p2)

        # Run equations
        dv1dt = ( (self.K1 * m2 * (p2-p1)) / p12**3 ) + ( (self.K1 * m3 * (p3-p1)) / p13**3 )
        dv2dt = ( (self.K1 * m1 * (p1-p2)) / p12**3 ) + ( (self.K1 * m3 * (p3-p2)) / p23**3 )
        dv3dt = ( (self.K1 * m1 * (p1-p3)) / p13**3 ) + ( (self.K1 * m2 * (p2-p3)) / p23**3 )
        dp1dt = self.K2 * v1
        dp2dt = self.K2 * v2
        dp3dt = self.K2 * v3

        # Package results to be returned to solver
        p12_derivs = np.concatenate((dp1dt, dp2dt))
        r_derivs = np.concatenate((p12_derivs, dp3dt))
        v12_derivs = np.concatenate((dv1dt, dv2dt))
        v_derivs = np.concatenate((v12_derivs, dv3dt))
        derivs = np.concatenate((r_derivs, v_derivs))
        return derivs
    
    
    def calculate_trajectories(self):
        """Calculates the trajectories of the 3 bodies in instance of problem"""
        print("Calculating trajectories...")
        s = time.perf_counter()
        # Prepare initial parameters
        init_params = np.array([self.p1, self.p2, self.p3, self.v1, self.v2, self.v3])
        init_params = init_params.flatten()
        time_span = np.linspace(0, self.orbital_periods, 750)

        # Run the odeint solver
        self.results = odeint(self.equations, init_params, time_span, args=(self.G, self.m1, self.m2, self.m3))

        e = time.perf_counter()
        print(f"Done ({ (e - s) * 1000} ms)")
        return self.results


    def animation_frame(self, i):
        """Function for animating display of data"""
        self.line1.set_data(self.p1_x[:i], self.p1_y[:i])
        self.line2.set_data(self.p2_x[:i], self.p2_y[:i])
        self.line3.set_data(self.p3_x[:i], self.p3_y[:i])
        return self.line1,self.line2,self.line3


    def display_trajectories(self, animated=True, save_animation=True):
        """Displays an animation or graph of results"""
        # Unpack results
        self.p1_x = self.results[:,0]
        self.p1_y = self.results[:,1]
        self.p2_x = self.results[:,2]
        self.p2_y = self.results[:,3]
        self.p3_x = self.results[:,4]
        self.p3_y = self.results[:,5]

        # Determine scale of window
        xmax = max([max(self.p1_x), max(self.p2_x), max(self.p3_x)])
        xmin = min([min(self.p1_x), min(self.p2_x), min(self.p3_x)])
        ymax = max([max(self.p1_y), max(self.p2_y), max(self.p3_y)])
        ymin = min([min(self.p1_y), min(self.p2_y), min(self.p3_y)])

        # Create visualization
        if animated:
            # Initial state of plot
            fig, ax = plt.subplots()
            ax.axis([xmin, xmax, ymin, ymax])
            self.line1, = ax.plot(self.p1[0], self.p1[1])
            self.line2, = ax.plot(self.p2[0], self.p2[1])
            self.line3, = ax.plot(self.p3[0], self.p3[1])

            # Animate the plot
            self.three_body_animation = FuncAnimation(fig, func=self.animation_frame, frames=range(len(self.p1_x)), interval=1, save_count=len(self.p1_x))

            plt.show()
        else:
            #Plot without animation
            fig, ax = plt.subplots()
            ax.plot(self.p1_x, self.p1_y)
            ax.plot(self.p2_x, self.p2_y)
            ax.plot(self.p3_x, self.p3_y)
            plt.show()

        if (animated and save_animation):
            # Save the animation
            print("Generating GIF...")
            s = time.perf_counter()
            self.three_body_animation.save('test_export.gif', writer=PillowWriter(fps=24))
            e = time.perf_counter()
            print(f"Done ({e - s} s)")


    def to_csv(self, filename):
        """Writes results to a csv file"""
        fields = ["p1_x", "p1_y", "p2_x", "p2_y", "p3_x", "p3_y", "v1_x", "v1_y", "v2_x", "v2_y", "v3_x", "v3_y"]     
        print(f"Writing to {filename}...")

        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerows(self.results)

        print("Done")

