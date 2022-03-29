# Compares the outputs for the classical solution and the ML solution to the four body problem
# Now with functionality for plotting center of mass

# Imports for the four body problem engine
from classical_solution_four_bodies import FourBodyProblem
import data_generator as data_gen

# Imports for ML
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow import keras

# Imports for other stuf
import numpy as np
import time
import csv
import math
from tqdm import tqdm
import random

def sortRawData(rCl, rMl):
    """Organizes data into formatted columns for easy analysis in an excel spreadsheet"""

    # Destination array
    sortedData = []

    # Loop through raw data from rCl and rMl arrays
    for x in range(6):
        # Temporary block variable will be appended to the destination array
        block = []
        for y in range(0,8,2):
            # x, y, and z are index variables
            z = y + 1

            # Calculate divergence before loading into an array
            # X coordinate difference
            xdiff = rCl[x,y]-rMl[x,y]
            # Y coordinate difference
            ydiff = rCl[x,z]-rMl[x,z]
            # Straight-line divergence with Pythagorean Theorem
            hdiff = math.sqrt(xdiff**2 + ydiff**2)

            # Load into temporary line array to be appended to block variable
            line = [rCl[x,y], rCl[x,z], rMl[x,y], rMl[x,z], xdiff, ydiff, hdiff]

            # Append line array to block variable
            if len(block) == 0:
                block = [line]
            else:
                block = np.append(block, [line], axis=0)
        
        # Append block array to destination array
        if len(sortedData) == 0:
            sortedData = block
        else:
            sortedData = np.append(sortedData, block, axis=1)

    return sortedData

def calcCOM(m, rCl, rMl):
    """Finds the center of mass"""

    # Classical solution x and y
    x_cl_com = (m[0]*rCl[0] + m[1]*rCl[2] + m[2]*rCl[4] + m[3]*rCl[6]) / (m[0] + m[1] + m[2] + m[3])
    y_cl_com = (m[0]*rCl[1] + m[1]*rCl[3] + m[2]*rCl[5] + m[3]*rCl[7]) / (m[0] + m[1] + m[2] + m[3])

    # ML solution x and y
    rMl = rMl[0]
    x_ml_com = (m[0]*rMl[0] + m[1]*rMl[2] + m[2]*rMl[4] + m[3]*rMl[6]) / (m[0] + m[1] + m[2] + m[3])
    y_ml_com = (m[0]*rMl[1] + m[1]*rMl[3] + m[2]*rMl[5] + m[3]*rMl[7]) / (m[0] + m[1] + m[2] + m[3])

    # Find divergence
    x_diff = x_cl_com - x_ml_com
    y_diff = y_cl_com - y_ml_com
    straightline_diff = math.sqrt(x_diff**2 + y_diff**2)

    # Return in array
    return [x_cl_com, y_cl_com, x_ml_com, y_ml_com, x_diff, y_diff, straightline_diff]

def calcEnergy(m, r):
    """
    Analyzes the change in energy over time
    m is masses, r is the positions
    """
    # Array for kinetic energies at each time interval
    xKE = []
    yKE = []
    
    # Loop through each time interval t and process each body b
    # x values
    for t in range(1, 6):
        total = 0
        for b in range(0, 4):
            velocity = ( r[t,b*2] - r[t-1,b*2] ) / 0.5
            kineticEnergy = (0.5) * m[b] * (velocity**2)
            if (velocity > 0):
                total+=kineticEnergy
            else:
                total-=kineticEnergy
        if (len(xKE) == 0):
            xKE = [total]
        else:
            xKE = np.append(xKE, total)

    # y values
    for t in range(1, 6):
        total = 0
        for b in range(0, 4):
            velocity = ( r[t,(b*2)+1] - r[t-1,(b*2)+1] ) / 0.5
            kineticEnergy = (0.5) * m[b] * (velocity**2)
            if (velocity > 0):
                total+=kineticEnergy
            else:
                total-=kineticEnergy
        if (len(yKE) == 0):
            yKE = [total]
        else:
            yKE = np.append(yKE, [total], axis=0)
    
    return np.append(xKE, yKE)


    

def getInput(fromTraining, trainingDataset):
    """Generates initial conditions, ensuring that they represent a solvable problem"""
    # Project-standard constants
    orbital_periods = 3
    time_steps = 1500
    
    if fromTraining:
        r = random.randrange(10_000)
        c = trainingDataset[r]
        inputs = [c[0], c[1], c[2], c[3], 
                [c[4], c[5]], [c[6], c[7]], [c[8], c[9]], [c[10], c[11]],
                [c[12], c[13]], [c[14], c[15]], [c[16], c[17]], [c[18], c[19]]
                ]
        problem = FourBodyProblem(orbital_periods, time_steps, c[0], c[1], c[2], c[3], [c[4], c[5]], [c[6], c[7]], [c[8], c[9]], [c[10], c[11]], [c[12], c[13]], [c[14], c[15]], [c[16], c[17]], [c[18], c[19]])
        s = time.perf_counter()
        output = problem.calculate_trajectories()
        e = time.perf_counter()
        classical_output = np.concatenate(([output[249,:8]], [output[499,:8]], [output[749,:8]], [output[999,:8]], [output[1249,:8]], [output[1499,:8]]), axis=0)
        
        computeTime = (e - s) * 1000
        return inputs, classical_output, computeTime
    else:
        

        output = []
        computeTime = 0

        # Find valid initial conditions; each problem must be checked to ensure the solver can handle it
        valid = False
        while not valid:
            # Generate a random problem
            m1 = data_gen.newMass()
            m2 = data_gen.newMass()
            m3 = data_gen.newMass()
            m4 = data_gen.newMass()
            p = data_gen.getPositions()
            v = data_gen.getVelocities()
            problem = FourBodyProblem(orbital_periods, time_steps, m1, m2, m3, m4, p[0], p[1], p[2], p[3], v[0], v[1], v[2], v[3])

            # Check problem to ensure that it is valid
            s = time.perf_counter()
            output = problem.calculate_trajectories()
            e = time.perf_counter()
            computeTime = (e - s) * 1000
            valid = data_gen.testValidity(output)

        inputs = [m1, m2, m3, m4, p[0], p[1], p[2], p[3], v[0], v[1], v[2], v[3]]
        classical_output = np.concatenate(([output[249,:8]], [output[499,:8]], [output[749,:8]], [output[999,:8]], [output[1249,:8]], [output[1499,:8]]), axis=0)
        print(str(len(classical_output)))
        
        return inputs, classical_output, computeTime


def assess(model, fromTraining, trainingDataset):
    """Compares the accuracy of the classical solution and the machine learning solution on one four-body problem"""

    # Define constants for this trial
    i, rCl, clComputeTime = getInput(fromTraining, trainingDataset)
    calculateToPeriod = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    trialCounter = 0

    # Raw Classical Output
    #rCl = []
    # Raw ML Output
    rMl = []
    # Compute times
    #computeTimes = []
    # Centers of mass
    com = []

    # Each trial compares the accuracy of each solution at six different times for the same initial conditions
    for period in calculateToPeriod:
        # Assess the classical solution
        print(f"Period {trialCounter}:")
        #classicalProblem = FourBodyProblem(period, 2, i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10], i[11])
        #s = time.perf_counter()
        #classicalResults = classicalProblem.calculate_trajectories()
        #e = time.perf_counter()
        #clComputeTime = (e - s) * 1000
        #classicalResults = classicalResults[-1,:8]

        # Append results of output for this time point to raw data array
        #if (len(rCl) == 0):
        #    rCl = [classicalResults]
        #else:
        #    rCl = np.append(rCl, [classicalResults], axis=0)
        
        # Assess the machine learning solution
        mlInput = [[period] + i[:4] + i[4] + i[5] + i[6] + i[7] + i[8] + i[9] + i[10] + i[11]]
        mlInput = np.array(mlInput, dtype='float32')
        #s = time.perf_counter()
        mlResults = model(mlInput)
        #e = time.perf_counter()
        #mlComputeTime = (e - s) * 1000

        # Append results of output for this time point to raw data array
        if (len(rMl) == 0):
            rMl = mlResults
        else:
            rMl = np.append(rMl, mlResults, axis=0)

        interval_com = calcCOM(i[:4], rCl[trialCounter], mlResults)
        if (len(com) == 0):
            com = interval_com
        else:
            com = np.append(com, interval_com)

        trialCounter += 1

        #periodComputeTime = np.array([clComputeTime, mlComputeTime], dtype='float32')
        #if (len(computeTimes) == 0):
        #    computeTimes = periodComputeTime
        #else:
        #    computeTimes = np.append(computeTimes, periodComputeTime)

    # Sort the data before returning it
    sorted = sortRawData(rCl, rMl)
    # Perform energy analysis
    eCl = calcEnergy(i[:4], rCl)
    eMl = calcEnergy(i[:4], rMl)
    energy = np.append(eCl, eMl)
    

    return (sorted, com, i, energy)


def main(numTrials, fromTraining=False):
    """Compares classical and ML solutions for specified number of trials"""

    # Create the model structure
    # Creating in main method increases performance because the same model can be used for multiple trials
    model = Sequential([
      Dense(256, activation=tf.keras.activations.tanh, input_shape=[21]),
      Dense(256, activation=tf.keras.activations.tanh),
      Dense(256, activation=tf.keras.activations.tanh),
      Dense(256, activation=tf.keras.activations.tanh),
      Dense(256, activation=tf.keras.activations.tanh),
      Dense(256, activation=tf.keras.activations.tanh),
      Dense(256, activation=tf.keras.activations.tanh),
      Dense(256, activation=tf.keras.activations.tanh),
      Dense(256, activation=tf.keras.activations.tanh),
      Dense(256, activation=tf.keras.activations.tanh),
      Dense(8),
])

    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])

    # Load the weights on the model
    model.load_weights("assets/tanh_checkpoints/cp.ckpt")

    # If testing on the training dataset, load the training dataset
    trainingDataset = []
    if fromTraining:
        print("Loading training data...")
        with open('outputs/master_dataset.csv', newline='') as masterFile:
            masterReader = csv.reader(masterFile, delimiter=',')
            for row in tqdm(masterReader):
                conditions = row[2:]
                conditions = np.asarray(conditions).astype("float32")

                if (len(trainingDataset) == 0):
                    trainingDataset = np.array([conditions], dtype='float32')
                else:
                    trainingDataset = np.append(trainingDataset, [conditions], axis=0)
            print(f"Rows: {len(trainingDataset)}")
            print(f"Columns: {len(trainingDataset[0])}")

    # Compare solutions
    results = []
    #performances = []
    com = []
    init = []
    energy = []
    for n in range(numTrials):
        print("")
        print("")
        print(f"Trial {n}:")
        print("")
        data = assess(model, fromTraining, trainingDataset)
        if len(results) == 0:
            results = data[0]
            #performances = [data[1]]
            com = [data[1]]
            init = [data[2]]
            energy = [data[3]]
        else:
            results = np.append(results, data[0], axis=0)
            #performances = np.append(performances, [data[1]], axis=0)
            com = np.append(com, [data[1]], axis=0)
            init = np.append(init, [data[2]], axis=0)
            energy = np.append(energy, [data[3]], axis=0)

    
    # Export results to a csv file
    with open("outputs/results.csv", 'w', newline='') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerows(results)
    #with open("outputs/performances.csv", 'w', newline='') as csvFile:
    #    csvWriter = csv.writer(csvFile)
    #    csvWriter.writerows(performances)
    with open("outputs/com.csv", 'w', newline='') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerows(com)
    with open("outputs/init_conditions.csv", 'w', newline='') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerows(init)
    with open("outputs/energy_analysis.csv", 'w', newline='') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerows(energy)


main(numTrials=1000, fromTraining=False)