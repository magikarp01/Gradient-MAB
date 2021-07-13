import numpy as np
import matplotlib
import math
import random
from scipy.spatial import ConvexHull
import gradDescent

SPSAObject = gradDescent.SPSA()
from tqdm import tqdm


# all params are in range [0, 1)
def randomParams(d):
    vec = [None]*d
    for i in range(d):
        vec[i] = random.random()
    return np.array(vec)


def uniformSPSABudget(f, k, d, maxBudget):
    budget = 0
    xPos = [None] * k  # current position
    xHats = [None] * k  # argmax over xPos so far
    fHats = [None] * k  # max values of f, corresponding to xHats

    for i in range(k):
        startPoint = randomParams(d)
        xPos[i] = startPoint
        xHats[i] = startPoint
        fHats[i] = f(startPoint)

    numRounds = int(maxBudget / k / 3)
    numEvals = k
    convergeDic = {}
    for round in range(numRounds):
        for i in range(k):
            partials = SPSAObject.partials(f, xPos[i], round)
            xPos[i] = SPSAObject.step(xPos[i], round, partials)
            fVal = f(xPos[i])
            if fVal > fHats[i]:
                fHats[i] = fVal
                xHats[i] = xPos[i]
            numEvals += 3
        convergeDic[numEvals] = max(fHats)

    maxIndex = np.argmax(fHats)
    return (xHats[maxIndex], fHats[maxIndex], convergeDic)

def uniformSPSABudgetGivenX(f, k, d, maxBudget, xPos):
    budget = 0
    xHats = [None] * k  # argmax over xPos so far
    fHats = [None] * k  # max values of f, corresponding to xHats

    for i in range(k):
        startPoint = randomParams(d)
        xHats[i] = startPoint
        fHats[i] = f(startPoint)

    numRounds = int(maxBudget / k / 3)
    numEvals = k
    convergeDic = {}
    for round in range(numRounds):
        for i in range(k):
            partials = SPSAObject.partials(f, xPos[i], round)
            xPos[i] = SPSAObject.step(xPos[i], round, partials)
            fVal = f(xPos[i])
            if fVal > fHats[i]:
                fHats[i] = fVal
                xHats[i] = xPos[i]
            numEvals += 3
        convergeDic[numEvals] = max(fHats)

    maxIndex = np.argmax(fHats)
    return (xHats[maxIndex], fHats[maxIndex], convergeDic)
