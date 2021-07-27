import numpy as np
import matplotlib
import math
import random
from scipy.spatial import ConvexHull
import gradDescent
import kriging

def randomParams(d):
    vec = [None]*d
    for i in range(d):
        vec[i] = random.random()
    return np.array(vec)


# takes negative differences
def getKroneckers(values):
    numInstances = len(values)

    minValue = values[0]
    for i in range(numInstances):
        if numInstances[i] < minValue:
            minValue = numInstances[i]

    kroneckers = [0]*numInstances
    for i in range(numInstances):
        # negative of usual, but since we're doing minimums it should work
        kroneckers[i] = values[i] - minValue

    return kroneckers


def getBudget(values, variances, kroneckers, numSamples):
    numInstances = len(values)

    optimalInstances = []
    for minIndex in range(numInstances):
        if kroneckers[minIndex] == 0:
            optimalInstances.append(minIndex)

    for instance in range(numInstances):
        if instance not in optimalInstances:
            if variances[instance] != 0 and kroneckers[instance] != 0:
                startInstance = instance
                break
    try:
        denom = variances[startInstance] / (kroneckers[startInstance] ** 2)
    except:  # happens when all of the nonoptimal moves have zero variance
        varFlag = False
    ratios = [0]*7

    for instance in range(numInstances):
        if instance in optimalInstances:
            continue
        elif instance == startInstance:
            ratios[instance] = 1
            continue
        else:
            ratio = variances[instance] / (kroneckers[instance] ** 2)
            try:
                ratio /= denom
            except:  # happens when all of the nonoptimal moves have zero variance, only one possible move
                budget = [0] * 7
                totBudget = 0
                for action in range(numInstances):
                    totBudget += numSamples[action]
                averageAlloc = (totBudget + 1) / len(optimalInstances)
                for x in optimalInstances:
                    budget[x] = averageAlloc

                return budget

            ratios[instance] = ratio


# set up calculation for the proportion of the budget for optimal actions
    tempSum = 0  # sum of N^2/stdev
    for action in range(numInstances):
        if ratios[action] != 0:
            tempSum += ratios[action] ** 2 / variances[action]  # stuff cancels
    tempSum = math.sqrt(tempSum)

    for action in optimalInstances:  # not sure if optimalProportion might be different for multiple optimal actions
        ratios[action] = math.sqrt(variances[action]) * tempSum

    # calculate the actual budgets from the proportions
    total = 0
    budget = [0]*7
    totBudget = 0
    for action in range(numInstances):
        total += ratios[action]
        totBudget += numSamples[action]

    try:
        scale = (totBudget + 1) / total
        for action in range(7):
            budget[action] = scale * ratios[action]
    except:
        uniform = (totBudget+1)/7
        for action in range(7):
            budget[action] = uniform

    return budget


def allocateSamples(budgetAlloc, batchSize):
    adjustedBudget = []
    totSize = sum(budgetAlloc)
    for i in range(len(budgetAlloc)):
        adjustedBudget.append(budgetAlloc[i]*batchSize/totSize)

    return


# minSamples will be done at the start so theres enough points to fit
def OCBA_SPSA_Budget(f, k, d, maxBudget, minSamples, batchSize):
    instances = [None]*k
    xHats = [None] * k
    fHats = [None] * k
    estMins = [None] * k
    variances = [None] * k
    kroneckers = [None] * k
    numSamples = [minSamples] * k


    finiteDifsObject = gradDescent.finiteDifs()
    for i in range(k):
        startPoint = randomParams(d)
        xHats[i] = startPoint
        fHats[i] = f(startPoint)
        # this is for minimizing not maximizing
        instances[i] = finiteDifsObject.gradDescent(f, startPoint, minSamples)[2]

    # change True to while budget < maxBudget
    while True:
        for i in range(k):
            points = []
            pointValues = []
            for point in instances[i]:
                points.append(point[0])
                pointValues.append(point[1])
            variances[i], estMins[i] = kriging.quadEstMin(points, pointValues)
        
        kroneckers = getKroneckers(estMins)
        budgetAlloc = getBudget(estMins, variances, kroneckers, numSamples)

# when fitting, we should make sure the function had a minimum
# or, if it is opposite, we should just have the value be unbounded
