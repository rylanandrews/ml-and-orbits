# Compares the outputs for the classical solution and the ML solution to the four body problem

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
    

def getInput(fromTraining, trainingDataset):
    """Generates initial conditions, ensuring that they represent a solvable problem"""
    
    if fromTraining:
        r = random.randrange(10_000)
        c = trainingDataset[r]
        return [c[0], c[1], c[2], c[3], 
                [c[4], c[5]], [c[6], c[7]], [c[8], c[9]], [c[10], c[11]],
                [c[12], c[13]], [c[14], c[15]], [c[16], c[17]], [c[18], c[19]]
                ]
    else:
        # Project-standard constants
        orbital_periods = 3
        time_steps = 1500

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
            output = problem.calculate_trajectories()
            valid = data_gen.testValidity(output)
        
        return [m1, m2, m3, m4, p[0], p[1], p[2], p[3], v[0], v[1], v[2], v[3]]


def assess(model, fromTraining, trainingDataset):
    """Compares the accuracy of the classical solution and the machine learning solution on one four-body problem"""

    # Define constants for this trial
    i = getInput(fromTraining, trainingDataset)
    calculateToPeriod = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    trialCounter = 1

    # Raw Classical Output
    rCl = []
    # Raw ML Output
    rMl = []
    # Compute times
    computeTimes = []

    # Each trial compares the accuracy of each solution at six different times for the same initial conditions
    for period in calculateToPeriod:
        # Assess the classical solution
        print(f"Period {trialCounter}:")
        classicalProblem = FourBodyProblem(period, 2, i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10], i[11])
        s = time.perf_counter()
        classicalResults = classicalProblem.calculate_trajectories()
        e = time.perf_counter()
        clComputeTime = (e - s) * 1000
        classicalResults = classicalResults[-1,:8]

        # Append results of output for this time point to raw data array
        if (len(rCl) == 0):
            rCl = [classicalResults]
        else:
            rCl = np.append(rCl, [classicalResults], axis=0)
        
        # Assess the machine learning solution
        mlInput = [[period] + i[:4] + i[4] + i[5] + i[6] + i[7] + i[8] + i[9] + i[10] + i[11]]
        mlInput = np.array(mlInput, dtype='float32')
        s = time.perf_counter()
        mlResults = model(mlInput)
        e = time.perf_counter()
        mlComputeTime = (e - s) * 1000

        # Append results of output for this time point to raw data array
        if (len(rMl) == 0):
            rMl = mlResults
        else:
            rMl = np.append(rMl, mlResults, axis=0)

        trialCounter += 1

        periodComputeTime = np.array([clComputeTime, mlComputeTime], dtype='float32')
        if (len(computeTimes) == 0):
            computeTimes = periodComputeTime
        else:
            computeTimes = np.append(computeTimes, periodComputeTime)

    # Sort the data before returning it
    return (sortRawData(rCl, rMl), computeTimes)


def main(numTrials, fromTraining=False):
    """Compares classical and ML solutions for specified number of trials"""

    # Create the model structure
    # Creating in main method increases performance because the same model can be used for multiple trials
    model = Sequential([
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=[21]),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.3),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.3),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.3),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.3),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dense(8),
    ])
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])

    # Load the weights on the model
    model.load_weights("assets/fusion_checkpoints/cp.ckpt")

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
    performances = []
    for n in range(numTrials):
        print("")
        print("")
        print(f"Trial {n}:")
        print("")
        data = assess(model, fromTraining, trainingDataset)
        if len(results) == 0:
            results = data[0]
            performances = [data[1]]
        else:
            results = np.append(results, data[0], axis=0)
            performances = np.append(performances, [data[1]], axis=0)
    
    # Export results to a csv file
    with open("temp1.csv", 'w', newline='') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerows(results)
    with open("temp2.csv", 'w', newline='') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerows(performances)


main(numTrials=100, fromTraining=False)