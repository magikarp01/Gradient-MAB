import numpy as np
import matplotlib.pyplot as plt
import math
import random
import gradDescent
from gradientAllocation import stratifiedSampling
from scipy.spatial import ConvexHull

SPSAObject = gradDescent.SPSA()
finiteDifsObject = gradDescent.finiteDifs()


# n is number of visits to the
# t is total number of calls to all instances
def h(n, t):
    expo = -n/math.sqrt(t)
    return math.exp(expo)


# finds convex hull with hSet as x, fSet as y
# h is from 0 to 1, f-hat is positive
# selects points which are on the convex hull, assuming all f are positive
def selectPoints(hSet, fSet):
    k = len(hSet)

    # point set shouldn't have duplicates
    naivePoints = [(hSet[i], fSet[i],) for i in range(k)]
    pointDic = {}
    for i in range(k):
        pointDic[naivePoints[i]] = i

    # shuffling makes it so duplicate removal is random i think
    random.shuffle(naivePoints)

    # list(set()) is supposed to remove duplicates, but it probably doesn't
    # it is possible that i don't need do list(set())
    points = list(set(naivePoints))

    getX = lambda point: point[0]
    pointsCopy = list(set(naivePoints))
    pointsCopy.sort(key = getX)
    # not enough points for convex hull
    if len(pointsCopy) <= 2:
        finIndices = []
        for point in pointsCopy:
            finIndices.append(pointDic[point])
        return finIndices

    # all points have same x
    if pointsCopy[0][0] == pointsCopy[-1][0]:
        maxPointIndex = 0
        maxY = pointsCopy[0][1]
        for pointIndex in range(len(pointsCopy)):
            if pointsCopy[pointIndex][1] > maxY:
                maxY = pointsCopy[pointIndex][1]
                maxPointIndex = pointIndex
        return [pointDic[pointsCopy[maxPointIndex]]]

    # should probably check that points are not collinear
    # almost certainly not collinear, i'm too lazy

    hull = ConvexHull(points).points.tolist()
    hull.sort(key = getX)
    minPoint = hull[0]
    maxPoint = hull[-1]

    slope = (maxPoint[1]-minPoint[1])/(maxPoint[0]-minPoint[0])
    # check if point lies above the line
    def checkThresh(point):
        xDif = point[0] - minPoint[0]
        # check if the y value is above the line
        return xDif*slope + minPoint[1] <= point[1]

    finIndices = []
    for point in hull:
        if checkThresh(point):
            finIndices.append(pointDic[tuple(point)])

    return finIndices


# selectedPoints is list of indices that should have samples allocated
# uniformly allocates the batch among the selected points
# batchSize should ideally be multiple of k
def allocateSamples(k, selectedPoints, batchSize):
    samplesPerInstance = batchSize // len(selectedPoints)

    results = [0]*k
    for i in selectedPoints:
        results[i] = samplesPerInstance

    residue = batchSize % len(selectedPoints)
    extras = random.sample(range(len(selectedPoints)), residue)
    for i in extras:
        results[selectedPoints[i]] += 1

    return results

# minSamples will be done at the start so theres enough points to fit
# assuming we want to minimize a function
# using finite differences for partials, not SPSA
# maxBudget must be much greater than k*minSamples*numEvalsPerGrad, min bound is k*(minSamples*numEvalsPerGrad + 1)
# k is number of instances
def metaMaxSearch(f, k, d, maxBudget, batchSize, numEvalsPerGrad, a=.02, c=.001):
    instances = [None] * k
    xHats = [None] * k
    fHats = [None] * k
    numSamples = [0] * k
    n = [1]*k

    finiteDifsObject = gradDescent.finiteDifs()
    elapsedBudget = 0

    startPositions = stratifiedSampling(d, k)

    for i in range(k):
        startPoint = startPositions[i]
        xHats[i] = startPoint
        fHats[i] = f(startPoint)
        elapsedBudget += 1
        numSamples[i] += 1
        instances[i] = (startPoint, fHats[i])


    # change True to while budget < maxBudget
    convergeDic = {}
    sampleDic = {}

    while elapsedBudget < maxBudget:
        hVals = [0]*k
        b = sum(n)
        for i in range(k):
            hVals[i] = h(n[i], b)
        selectedPoints = selectPoints(hVals, fHats)
        sampleAlloc = allocateSamples(k, selectedPoints, batchSize)

        # perform sampleAlloc[i] steps for every instance
        # could add in multi-threading here
        for i in range(k):
            samples = sampleAlloc[i]
            for j in range(samples):
                n[i] += 1

                # step from the previous point of the ith instance once
                partials = finiteDifsObject.partials(f, instances[i][-1][0], numSamples[i])
                newX = finiteDifsObject.step(instances[i][-1][0], numSamples[i], partials)
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

"""
# f is loss function to be evaluated
# k is number of instances to run
# d is num of dimensions
# budget is number that can be allocated
# this metaMax gets the maximum, not the minimum
def metaMaxSPSA(f, k, d, numRounds):

    n = [1] * k
    xPos = [None] * k # current position
    xHats = [None]*k # argmax over xPos so far
    fHats = [None]*k # max values of f, corresponding to xHats

    for i in range(k):
        startPoint = randomParams(d)
        xPos[i] = startPoint
        xHats[i] = startPoint
        fHats[i] = f(startPoint)

    # budget = t_r, function has already been evaluated k times
    budget = k

    for round in tqdm(range(numRounds)):
        # default is that the instance was not sampled

        # find the hVals for this round
        hVals = [None]*k
        for i in range(k):
            hVals[i] = h(n[i], budget)

        # select the points for sampling
        selected = selectPoints(hVals, fHats)

        # update the selected points

        for pointIndex in selected:
            partials = np.negative(SPSAObject.partials(f, xPos[pointIndex], round))
            xPos[pointIndex] = SPSAObject.step(xPos[pointIndex], round, partials)
            n[pointIndex] += 1
            fVal = f(xPos[pointIndex])
            if fVal > fHats[pointIndex]:
                fHats[pointIndex] = fVal
                xHats[pointIndex] = xPos[pointIndex]
            # possibly budget += 1

        budget += 3*len(selected)

    maxIndex = np.argmax(fHats)
    return (xHats[maxIndex], fHats[maxIndex], n, budget)
    
"""

