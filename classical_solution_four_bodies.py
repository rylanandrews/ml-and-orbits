# Rylan Andrews
# 10/14/2020
# Four body problem solver engine

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

class FourBodyProblem:
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


    def __init__(self, orbital_periods, time_steps, m1, m2, m3, m4, p1, p2, p3, p4, v1, v2, v3, v4):
        """Initialize FourBodyProblem object"""
        self.orbital_periods = orbital_periods
        self.time_steps = time_steps
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.v4 = v4

        # Convert position vectors to arrays
        p1 = np.array(p1, dtype="float64")
        p2 = np.array(p2, dtype="float64")
        p3 = np.array(p3, dtype="float64")
        p4 = np.array(p4, dtype="float64")

        # Convert velocity vectors to arrays
        v1 = np.array(v1, dtype="float64")
        v2 = np.array(v2, dtype="float64")
        v3 = np.array(v3, dtype="float64")
        v4 = np.array(v4, dtype="float64")


    def equations(self, w, t, G, m1, m2, m3, m4):
        """Equation model to be passed to odeint solver"""
        # Unpack data
        p1 = w[:2]
        p2 = w[2:4]
        p3 = w[4:6]
        p4 = w[6:8]
        v1 = w[8:10]
        v2 = w[10:12]
        v3 = w[12:14]
        v4 = w[14:16]

        # Find distance between bodies
        p12 = sci.linalg.norm(p2-p1)
        p13 = sci.linalg.norm(p3-p1)
        p14 = sci.linalg.norm(p4-p1)
        p23 = sci.linalg.norm(p3-p2)
        p24 = sci.linalg.norm(p4-p2)
        p34 = sci.linalg.norm(p4-p3)

        # Run equations
        dv1dt = ( (self.K1 * m2 * (p2-p1)) / p12**3 ) + ( (self.K1 * m3 * (p3-p1)) / p13**3 ) + ( (self.K1 * m4 * (p4-p1)) / p14**3 )
        dv2dt = ( (self.K1 * m1 * (p1-p2)) / p12**3 ) + ( (self.K1 * m3 * (p3-p2)) / p23**3 ) + ( (self.K1 * m4 * (p4-p2)) / p24**3 )
        dv3dt = ( (self.K1 * m1 * (p1-p3)) / p13**3 ) + ( (self.K1 * m2 * (p2-p3)) / p23**3 ) + ( (self.K1 * m4 * (p4-p3)) / p34**3 )
        dv4dt = ( (self.K1 * m1 * (p1-p4)) / p14**3 ) + ( (self.K1 * m2 * (p2-p4)) / p24**3 ) + ( (self.K1 * m3 * (p3-p4)) / p34**3 )
        dp1dt = self.K2 * v1
        dp2dt = self.K2 * v2
        dp3dt = self.K2 * v3
        dp4dt = self.K2 * v4

        # Package results to be returned to solver
        p12_derivs = np.concatenate((dp1dt, dp2dt))
        p123_derivs = np.concatenate((p12_derivs, dp3dt))
        p_derivs = np.concatenate((p123_derivs, dp4dt))
        v12_derivs = np.concatenate((dv1dt, dv2dt))
        v123_derivs = np.concatenate((v12_derivs, dv3dt))
        v_derivs = np.concatenate((v123_derivs, dv4dt))
        derivs = np.concatenate((p_derivs, v_derivs))
        return derivs
    
    
    def calculate_trajectories(self):
        """Calculates the trajectories of the 3 bodies in instance of problem"""
        print("Calculating trajectories...")
        s = time.perf_counter()
        # Prepare initial parameters
        init_params = np.array([self.p1, self.p2, self.p3, self.p4, self.v1, self.v2, self.v3, self.v4])
        init_params = init_params.flatten()
        self.time_span = np.linspace(0, self.orbital_periods, self.time_steps)

        # Run the odeint solver
        self.results = odeint(self.equations, init_params, self.time_span, args=(self.G, self.m1, self.m2, self.m3, self.m4))

        e = time.perf_counter()
        print(f"Done ({ (e - s) * 1000} ms)")
        return self.results


    def animation_frame(self, i):
        """Function for animating display of data"""
        self.line1.set_data(self.p1_x[:i], self.p1_y[:i])
        self.line2.set_data(self.p2_x[:i], self.p2_y[:i])
        self.line3.set_data(self.p3_x[:i], self.p3_y[:i])
        self.line4.set_data(self.p4_x[:i], self.p4_y[:i])
        return self.line1,self.line2,self.line3,self.line4


    def display_trajectories(self, animated=True, save_animation=True):
        """Displays an animation or graph of results"""
        # Unpack results
        self.p1_x = self.results[:,0]
        self.p1_y = self.results[:,1]
        self.p2_x = self.results[:,2]
        self.p2_y = self.results[:,3]
        self.p3_x = self.results[:,4]
        self.p3_y = self.results[:,5]
        self.p4_x = self.results[:,6]
        self.p4_y = self.results[:,7]

        # Determine scale of window
        xmax = max([max(self.p1_x), max(self.p2_x), max(self.p3_x), max(self.p4_x)])
        xmin = min([min(self.p1_x), min(self.p2_x), min(self.p3_x), min(self.p4_x)])
        ymax = max([max(self.p1_y), max(self.p2_y), max(self.p3_y), max(self.p4_y)])
        ymin = min([min(self.p1_y), min(self.p2_y), min(self.p3_y), min(self.p4_y)])

        # Create visualization
        if animated:
            # Initial state of plot
            fig, ax = plt.subplots()
            ax.axis([xmin, xmax, ymin, ymax])
            self.line1, = ax.plot(self.p1[0], self.p1[1])
            self.line2, = ax.plot(self.p2[0], self.p2[1])
            self.line3, = ax.plot(self.p3[0], self.p3[1])
            self.line4, = ax.plot(self.p4[0], self.p4[1])

            # Animate the plot
            self.three_body_animation = FuncAnimation(fig, func=self.animation_frame, frames=range(len(self.p1_x)), interval=1, save_count=len(self.p1_x))

            plt.show()
        else:
            #Plot without animation
            fig, ax = plt.subplots()
            ax.plot(self.p1_x, self.p1_y)
            ax.plot(self.p2_x, self.p2_y)
            ax.plot(self.p3_x, self.p3_y)
            ax.plot(self.p4_x, self.p4_y)
            plt.show()

        if (animated and save_animation):
            # Save the animation
            print("Generating GIF...")
            s = time.perf_counter()
            self.three_body_animation.save('test_export.gif', writer=PillowWriter(fps=24))
            e = time.perf_counter()
            print(f"Done ({e - s} s)")


    def to_csv(self, filename, withHeader=True):
        """Writes results to a csv file"""
        # Create header array
        header = ["Time", "x Position 1", "y Position 1", "x Position 2", "y Position 2", "x Position 3", "y Position 3", "x Position 4", "y Position 4", "x Velocity 1", "y Velocity 1", "x Velocity 2", "y Velocity 2", "x Velocity 3", "y Velocity 3", "x Velocity 4", "y Velocity 4"]     
        print(f"Writing to {filename}...")

        # Add times to the results array
        times = np.array(self.time_span)
        times = times[:,None]
        fileData = np.concatenate((times, self.results), axis=1)

        # Write the fileData array to a CSV file
        with open(filename, 'w', newline='') as csvFile:
            csvWriter = csv.writer(csvFile)
            if withHeader:
                csvWriter.writerow(header)
            csvWriter.writerows(fileData)

        print("Done")

