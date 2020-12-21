from classical_solution_v4 import FourBodyProblem
import numpy as np
import random
import csv
import time

def newMass():
    """Generates a random, non-zero mass between 0 and 2"""
    m = 0
    while (m == 0):
        m = random.random() * 2

    return m

def newPos():
    n = random.random() + 0.5
    if (random.random() > 0.5):
        n = n * -1
    
    return n

def getPositions():
    """Generates a set of positions, ensuring points are at least 0.5 units apart and no more than 1.5 units apart"""
    # First point is in range -0.5 to 0.5
    p1_x = random.random() - 0.5
    p1_y = random.random() - 0.5
    
    # Each successive point is added to previous point
    p2_x = p1_x + newPos()
    p2_y = p1_y + newPos()

    p3_x = p2_x + newPos()
    p3_y = p2_y + newPos()

    p4_x = p3_x + newPos()
    p4_y = p3_y + newPos()

    # Package results
    positions = [[p1_x, p1_y],
                 [p2_x, p2_y],
                 [p3_x, p3_y],
                 [p4_x, p4_y]]

    return positions

def newV():
    """Returns a velocity between -0.05 and 0.05"""
    return random.random() * 0.1 - 0.05

def getVelocities():
    """Creates 4 velocity vectors"""
    velocities = [[newV(), newV()],
                  [newV(), newV()],
                  [newV(), newV()],
                  [newV(), newV()]]

    return velocities

def testValidity(data):
    isValid = True
    counter = 0

    while isValid and (counter < 1499):
        diff = abs(data[counter + 1,0] - data[counter,0])
        if (diff > 0.5):
            isValid = False 

        counter += 1
    
    return isValid

def createData(numProblems, time_steps, withAnimation, withHeader):
    orbital_periods = 3
    numInvalid = 0
    totalCalcTime = 0

    with open("outputs/master.csv", 'w', newline='') as csvFile:
        header = ["Trial", "Path", "Mass 1", "Mass 2", "Mass 3", "Mass 4", "x Position 1", "y Position 1", "x Position 2", "y Position 2", "x Position 3", "y Position 3", "x Position 4", "y Position 4", "x Velocity 1", "y Velocity 1", "x Velocity 2", "y Velocity 2", "x Velocity 3", "y Velocity 3", "x Velocity 4", "y Velocity 4"]
        csvWriter = csv.writer(csvFile)
        if withHeader:
            csvWriter.writerow(header)

    with open("outputs/dataset.csv", 'w', newline='') as csvFile:
        header = ["Time", "x Position 1", "y Position 1", "x Position 2", "y Position 2", "x Position 3", "y Position 3", "x Position 4", "y Position 4", "x Velocity 1", "y Velocity 1", "x Velocity 2", "y Velocity 2", "x Velocity 3", "y Velocity 3", "x Velocity 4", "y Velocity 4"]
        csvWriter = csv.writer(csvFile)
        if withHeader:
            csvWriter.writerow(header)

    dataset = []

    for n in range(numProblems):
        print("Trial " + str(n+1))
        calcTime = 0
        valid = False
        
        while not valid:
            # Generate random masses in range 0 - 2
            m1 = newMass()
            m2 = newMass()
            m3 = newMass()
            m4 = newMass()

            p = getPositions()

            v = getVelocities()

            problem = FourBodyProblem(orbital_periods, time_steps, m1, m2, m3, m4, p[0], p[1], p[2], p[3], v[0], v[1], v[2], v[3])

            s = time.perf_counter()
            output = problem.calculate_trajectories()
            e = time.perf_counter()
            calcTime = e - s

            valid = testValidity(output)

            if not valid:
                print("Solution invalid, recalculating...")
                numInvalid += 1

        totalCalcTime += calcTime

        if withAnimation:
            problem.display_trajectories(animated=True, save_animation=False)

        problem.to_csv("outputs/num" + str(n) + ".csv", withHeader=withHeader)

        problemArgs = [str(n+1), "num" + str(n) + ".csv", str(m1), str(m2), str(m3), str(m4), str(p[0][0]), str(p[0][1]), str(p[1][0]), str(p[1][1]), str(p[2][0]), str(p[2][1]), str(p[3][0]), str(p[3][1]), str(v[0][0]), str(v[0][1]), str(v[1][0]), str(v[1][1]), str(v[2][0]), str(v[2][1]), str(v[3][0]), str(v[3][1])]

        with open('outputs/master.csv', 'a', newline='') as csvFile:
            csvWriter = csv.writer(csvFile)
            csvWriter.writerow(problemArgs)

        time_span = np.linspace(0, orbital_periods, time_steps)
        times = np.array(time_span)
        times = times[:,None]
        outputWithTime = np.concatenate((times, output), axis=1)

        with open('outputs/dataset.csv', 'a', newline='') as csvFile:
            csvWriter = csv.writer(csvFile)
            csvWriter.writerows(outputWithTime)

        if (len(dataset) == 0):
            dataset = output[:,0:8]
        else:
            dataset = np.append(dataset, output[:,0:8], axis=0)

    print(dataset[0])

    np.save("outputs/trainingOutputs.npy", dataset)

    avgCalcTime = ( totalCalcTime / numProblems ) * 1000

    print()
    print("Done Generating Solutions")
    print("=========================")
    print("Number Generated: " + str(numProblems))
    print("Number Invalid: " + str(numInvalid))
    print("Total Calculation Time (s): " + str(totalCalcTime))
    print("Avg Calculation Time (ms): " + str(avgCalcTime))

createData(200, time_steps=1500, withAnimation=False, withHeader=False)