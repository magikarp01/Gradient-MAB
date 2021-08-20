import numpy as np
import matplotlib.pyplot as plt
import math
import random
import gradDescent
import kriging
import pyDOE
import functions
from gradientAllocation import stratifiedSampling


# takes negative differences
def getKroneckers(values):
    numInstances = len(values)

    minValue = values[0]
    for i in range(numInstances):
        if values[i] < minValue:
            minValue = values[i]

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

# allocate samples given a budget allocation with fractions and a whole number batch size
# returns array of integers
def allocateSamples(budgetAlloc, batchSize):
    totSize = sum(budgetAlloc)
    fracParts = {}
    intParts = []
    for i in range(len(budgetAlloc)):
        est = budgetAlloc[i]*batchSize/totSize
        intParts.append(math.floor(est))
        fracParts[est-intParts[i]] = i

    residue = int(round(batchSize - sum(intParts)))
    sortedFracParts = sorted(fracParts.keys(), reverse=True)

    for r in range(residue):
        intParts[fracParts[sortedFracParts[r]]] += 1


    return intParts


# minSamples will be done at the start so theres enough points to fit
# assuming we want to minimize a function
# using finite differences for partials, not SPSA
# maxBudget must be much greater than k*minSamples*numEvalsPerGrad, min bound is k*(minSamples*numEvalsPerGrad + 1)
# k is number of instances
def OCBASearch(f, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples, discountRate=1, a=.001, c=.001):
    instances = [None]*k
    xHats = [None] * k
    fHats = [None] * k
    estMins = [None] * k
    variances = [None] * k
    numSamples = [0] * k

    finiteDifsObject = gradDescent.finiteDifs()
    elapsedBudget = 0

    startPositions = stratifiedSampling(d, k)

    for i in range(k):
        startPoint = startPositions[i]
        xHats[i] = startPoint
        fHats[i] = f(startPoint)
        elapsedBudget += 1
        numSamples[i] += 1

        # this is for minimizing not maximizing
        instances[i] = finiteDifsObject.gradDescent(f, startPoint, minSamples, a, c)[2]
        elapsedBudget += minSamples * numEvalsPerGrad
        numSamples[i] += minSamples * numEvalsPerGrad

    # change True to while budget < maxBudget
    convergeDic = {}
    sampleDic = {}
    while elapsedBudget < maxBudget:
        for i in range(k):
            points = []
            pointValues = []
            for point in instances[i]:
                # each point is (xValue, fValue)
                points.append(point[0])
                pointValues.append(point[1])


            estMins[i], variances[i] = kriging.quadEstMin(points, pointValues, discountRate)
        
        kroneckers = getKroneckers(estMins)
        budgetAlloc = getBudget(estMins, variances, kroneckers, numSamples)
        # sample allocation is the actual allocations to give to each
        sampleAlloc = allocateSamples(budgetAlloc, batchSize)

        # perform sampleAlloc[i] steps for every instance
        # could add in multi-threading here
        for i in range(k):
            samples = sampleAlloc[i]
            for j in range(samples):
                # step from the previous point of the ith instance once

                partials = finiteDifsObject.partials(f, instances[i][-1][0], numSamples[i], c=c)
                newX = finiteDifsObject.step(instances[i][-1][0], numSamples[i], partials, a=a)
                instances[i].append((newX, f(newX)))
                elapsedBudget += numEvalsPerGrad + 1

                fVal = f(instances[i][-1])
                elapsedBudget += 1
                if fVal < fHats[i]:
                    fHats[i] = fVal
                    xHats[i] = instances[i][-1]

                numSamples[i] += numEvalsPerGrad + 1
        convergeDic[elapsedBudget] = min(fHats)
        sampleDic[elapsedBudget] = numSamples.copy()
    maxIndex = np.argmax(fHats)
    return (xHats[maxIndex], fHats[maxIndex], convergeDic, instances, numSamples, sampleDic)


