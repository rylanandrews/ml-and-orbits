import numpy as np
import csv

trainingInputs = []
counter = 0
trainPath = "C:\\Users\\Rylan\\Documents\\Schoolwork\\12th Grade\\Science Research 2021\\Programs\\assets\\"

# Recreate the time points for the simulation
time_span = np.linspace(0, 3, 1500)
times = np.array(time_span)
times = times[:,None]

print("Creating trainingInputs array...")
# Use master csv file to create the training input array
with open(trainPath + 'master.csv', newline='') as masterFile:
  masterReader = csv.reader(masterFile, delimiter=',')
  # Loop through each row in master file
  for masterRow in masterReader:
    # Since there are 1500 data points per solution, create an array with 1500 rows of initial conditions
    initConditions = np.full((1500, 20), np.asarray(masterRow[2:]).astype('float32'))
    # Add the times to array
    inputs = np.concatenate((times, initConditions), axis=1)
    # Append resulting array to training inputs array
    if (len(trainingInputs) == 0):
      trainingInputs = inputs
    else:
      trainingInputs = np.append(trainingInputs, inputs, axis=0)
    counter += 1
    print(str(counter))

print(len(trainingInputs))
print(len(trainingInputs[0]))
print(str(trainingInputs[0]))

np.save("outputs/master.npy", trainingInputs)