import re

import functions
import pickle

class parameters:
    def __init__(self, numProcesses, iterPerProcess, sharedParams):
        self.numProcesses = numProcesses
        self.iterPerProcess = iterPerProcess
        self.sharedParams = sharedParams

    def load(self, loc):
        with open(loc, "rb") as h:
            savedParams = pickle.load(h)
            self.numProcesses = savedParams.numProcesses
            self.iterPerProcess = savedParams.iterPerProcess
            self.sharedParams = savedParams.sharedParams

    def save(self, path):
        with open(path + "/params.txt", "wb") as g:
            pickle.dump(self, g)


    def display(self):
        print(f"Number of processes is {self.numProcesses}")
        print(f"Iterations per process is {self.iterPerProcess}")
        print("Shared parameters: ", end="")
        print(self.sharedParams)

def readParams(loc):
    g = open(loc, 'r')
    lines = g.readlines()
    valDic = {}
    for line in lines:
        if line.startswith("#") or line=="\n":
            continue

        line = line.strip(' \t\n\r')
        line = line.replace(" ", "")
        keyName = line[:line.index('=')]
        val = line[line.index('=')+1:]
        valDic[keyName] = val

    numProcesses = int(valDic["numProcesses"])
    iterPerProcess = int(valDic["iterPerProcess"])

    if valDic["fun"] == "functions.griewank_adjusted":
        fun = functions.griewank_adjusted
    elif valDic["fun"] == "functions.ackley_adjusted":
        fun = functions.ackley_adjusted
    elif valDic["fun"] == "functions.rastrigin_adjusted":
        fun = functions.rastrigin_adjusted
    elif valDic["fun"] == "functions.random_ackley":
        fun = functions.random_ackley
    elif valDic["fun"] == "functions.random_griewank":
        fun = functions.random_griewank
    elif valDic["fun"] == "functions.random_rastrigin":
        fun = functions.random_rastrigin

    k = int(valDic["k"])
    d = int(valDic["d"])
    maxBudget = int(valDic["maxBudget"])
    allocSize = int(valDic["allocSize"])
    batchSize = int(valDic["batchSize"])
    numEvalsPerGrad = int(valDic["numEvalsPerGrad"])
    minSamples = int(valDic["minSamples"])

    a = float(valDic["a"])
    c = float(valDic["c"])
    useSPSA = (valDic["useSPSA"] == "True")
    discountFactor = float(valDic["discountFactor"])
    slidingWindow = int(valDic["slidingWindow"])

    minimum = float(valDic["minimum"])
    randomPos = (valDic["randomPos"] == "True")


    params = [fun, k, d, maxBudget, allocSize, batchSize, numEvalsPerGrad, minSamples, a, c, useSPSA,
              discountFactor, slidingWindow, minimum]

    return numProcesses, iterPerProcess, params, randomPos



# paramObject = parameters(numProcesses, iterPerProcess, params)
# paramObject.save(path)
#
# newParamObject = parameters(None, None, None)
# newParamObject.load(path + "/params.txt")
# newParamObject.display()

