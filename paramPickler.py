import re

import functions
import pickle

class parameters:
    def __init__(self, numProcesses, iterPerProcess, sharedParams):
        self.numProcesses = numProcesses
        self.iterPerProcess = iterPerProcess
        self.sharedParams = sharedParams

    def load(self, loc):
        h = open(loc, "rb")
        savedParams = pickle.load(h)
        self.numProcesses = savedParams.numProcesses
        self.iterPerProcess = savedParams.iterPerProcess
        self.sharedParams = savedParams.sharedParams

    def save(self, path):
        g = open(path + "/params.txt", "wb")
        pickle.dump(self, g)


    def display(self):
        print(f"Number of processes is {self.numProcesses}")
        print(f"Iterations per process is {self.iterPerProcess}")
        print("Shared parameters: ", end="")
        print(self.sharedParams)

def readParams(loc):
    g = open(loc, 'r')
    lines = g.readlines()
    newLines = []
    for line in lines:
        if line.startswith("#") or line=="\n":
            continue
        newLines.append(line[line.index('=')+2:].strip(' \t\n\r'))
    lines = newLines
    numProcesses = int(lines[0])
    iterPerProcess = int(lines[1])

    if lines[2] == "functions.griewank_adjusted":
        fun = functions.griewank_adjusted
    elif lines[2] == "functions.ackley_adjusted":
        fun = functions.ackley_adjusted

    k = int(lines[3])
    d = int(lines[4])
    maxBudget = int(lines[5])
    batchSize = int(lines[6])
    numEvalsPerGrad = int(lines[7])
    minSamples = int(lines[8])

    minimum = float(lines[9])
    discountRate = float(lines[10])
    a = float(lines[11])
    c = float(lines[12])
    useSPSA = (lines[13] == "True")
    discountFactor = float(lines[14])
    slidingWindow = int(lines[15])
    randomPos = (lines[16] == "True")

    params = [fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples, minimum, discountRate, a, c, useSPSA,
              discountFactor, slidingWindow]

    return numProcesses, iterPerProcess, params, randomPos



# paramObject = parameters(numProcesses, iterPerProcess, params)
# paramObject.save(path)
#
# newParamObject = parameters(None, None, None)
# newParamObject.load(path + "/params.txt")
# newParamObject.display()

