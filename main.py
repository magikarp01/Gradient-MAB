import random

import gradDescent
import functions
import metaMax
import uniform
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np

def compareGraphs(f, k, d, budget):
    xPos = []
    for i in range(k):
        xPos.append(metaMax.randomParams(d))

    metaMaxResults = metaMax.SPSABudgetGivenX(f, k, d, budget, xPos)
    uniformResults = uniform.uniformSPSABudgetGivenX(f, k, d, budget, xPos)
    print(metaMaxResults)
    plt.plot(*zip(*sorted(metaMaxResults[2].items())), label="metaMax")
    plt.plot(*zip(*sorted(uniformResults[2].items())), label="uniform")
    plt.legend
    plt.legend()
    plt.show()

# compareGraphs(functions.reverse_ackley_adjusted, 100, 5, 200000)

# functions.display3D(functions.reverse_ackley_adjusted, [0,1])

def compareAverages(f, numRuns, d, k, budget):
    metaMaxVals = []
    uniformVals = []
    metaMaxTimes = []
    uniformTimes = []

    for iteration in tqdm(range(numRuns)):
        mTStart = time.time()
        mV = metaMax.SPSABudget(f, k, d, budget)
        mT = time.time()-mTStart
        uTStart = time.time()
        uV = uniform.SPSABudget(f, k, d, budget)
        uT = time.time() - uTStart
        metaMaxVals.append(mV[1])
        uniformVals.append(uV[1])
        metaMaxTimes.append(mT)
        uniformTimes.append(uT)

    print(metaMaxVals)
    print(sum(metaMaxVals) / 100)
    print()

    print(uniformVals)
    print(sum(uniformVals) / 100)
    print()

    print(metaMaxTimes)
    print(sum(metaMaxTimes) / 100)
    print()

    print(uniformTimes)
    print(sum(uniformTimes) / 100)

# compareAverages(functions.reverse_ackley_adjusted, )


# x = metaMax.randomParams(3)
# f = functions.ackley_adjusted
# initVal = f(x)
# (minParams, min) = SPSA.gradDescent(f, x, 3, 10000, 1000, 200)

# print(metaMax.SPSABudget(functions.reverse_ackley_adjusted, 1000, 10, 20000))

SPSAObject = gradDescent.SPSA()
finiteDifsObject = gradDescent.finiteDifs()
testF = lambda x: (x[0]**4 - 3*x[0]**3 - 9*x[0]**2 + 1) + (x[1]**4 - 3*x[1]**3 - 9*x[1]**2 + 1)



# params = metaMax.randomParams(10)
# print(params)
# print(functions.reverse_ackley_adjusted(params))
# min = finiteDifsObject.gradAscent(functions.reverse_ackley_adjusted, params, 10000,a=.1)
# min = finiteDifsObject.gradDescent(testF, np.array([-1, -1]), 10000)
min = metaMax.finiteDifsBudget(functions.reverse_ackley_adjusted, 100, 5, 100000, a=.01)
for r in min:
    print(r)
