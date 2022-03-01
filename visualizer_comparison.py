from classical_solution_four_bodies import FourBodyProblem
import numpy as np
import random
import csv
import time
import scipy as sci
import numpy as np
from scipy.integrate import odeint
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow import keras
import matplotlib.pyplot as plt 
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation

"""
def animation_frame(i):
        """"""Function for animating display of data""""""
        line1.set_data(p1_x[:i], p1_y[:i])
        line2.set_data(p2_x[:i], p2_y[:i])
        line3.set_data(p3_x[:i], p3_y[:i])
        line4.set_data(p4_x[:i], p4_y[:i])
        return line1,line2,line3,line4
"""

def display_trajectories(results, animated=True, save_animation=True):
    """Displays an animation or graph of results"""
    # Unpack results
    p1_x = results[:,0]
    p1_y = results[:,1]
    p2_x = results[:,2]
    p2_y = results[:,3]
    p3_x = results[:,4]
    p3_y = results[:,5]
    p4_x = results[:,6]
    p4_y = results[:,7]

    # Determine scale of window
    xmax = max([max(p1_x), max(p2_x), max(p3_x), max(p4_x)])
    xmin = min([min(p1_x), min(p2_x), min(p3_x), min(p4_x)])
    ymax = max([max(p1_y), max(p2_y), max(p3_y), max(p4_y)])
    ymin = min([min(p1_y), min(p2_y), min(p3_y), min(p4_y)])

    # Create visualization
    if animated:
        # Initial state of plot
        fig, ax = plt.subplots()
        ax.axis([-3.75, 0.6, -0.52, 3.5])
        #line1, = ax.plot(p1[0], p1[1])
        #line2, = ax.plot(p2[0], p2[1])
        #line3, = ax.plot(p3[0], p3[1])
        #line4, = ax.plot(p4[0], p4[1])

        # Animate the plot
        three_body_animation = FuncAnimation(fig, func=animation_frame, frames=range(len(p1_x)), interval=1, save_count=len(p1_x))

        plt.show()
    else:
        #Plot without animation
        fig, ax = plt.subplots()
        ax.axis([-3.75, 0.6, -0.52, 3.5])
        ax.plot(p1_x, p1_y)
        ax.plot(p2_x, p2_y)
        ax.plot(p3_x, p3_y)
        ax.plot(p4_x, p4_y)
        plt.show()

    if (animated and save_animation):
        # Save the animation
        print("Generating GIF...")
        s = time.perf_counter()
        three_body_animation.save('test_export.gif', writer=PillowWriter(fps=24))
        e = time.perf_counter()
        print(f"Done ({e - s} s)")

def main():
    classical_solution = FourBodyProblem(3, 1000, 1.493274915, 0.777043742, 0.69905188, 1.644425625, [-0.2235866560691493, 0.4662489456089146], [0.45460058216449073, 1.856775210402708], [-0.581630066079448, 2.652030222251113], [-1.0887887921718722, 1.8916735993939726], [-0.001983479092496618, -0.023803719445120764], [-0.04168724914664399, -0.022727286512001968], [-0.0006659848704990909, -0.01954219193498398], [0.0222475650915121, -0.019197877602321424])

    classical_solution.calculate_trajectories()

    classical_solution.display_trajectories(animated=False, save_animation=False)


    model = Sequential([
        Dense(256, activation='relu', input_shape=[21]),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(8),
        ])

    model.compile(optimizer='SGD', loss='mean_squared_error', metrics=['accuracy'])

    # Load the weights on the model
    model.load_weights("assets/sgd_checkpoints/cp.ckpt")

    time_span = np.linspace(0, 3, 1000)

    ml_outputs = []

    for n in time_span:
        ml_input = [n, 1.493274915, 0.777043742, 0.69905188, 1.644425625, -0.2235866560691493, 0.4662489456089146, 0.45460058216449073, 1.856775210402708, -0.581630066079448, 2.652030222251113, -1.0887887921718722, 1.8916735993939726, -0.001983479092496618, -0.023803719445120764, -0.04168724914664399, -0.022727286512001968, -0.0006659848704990909, -0.01954219193498398, 0.0222475650915121, -0.019197877602321424]
        ml_input = np.array([ml_input], dtype='float32')
        mlResults = model(ml_input)
        if (len(ml_outputs) == 0):
            ml_outputs = mlResults
        else:
            ml_outputs = np.append(ml_outputs, mlResults, axis=0)

    print(str(ml_outputs))
    print(str(len(ml_outputs[0])))

    display_trajectories(ml_outputs, animated=False, save_animation=False)

main()

