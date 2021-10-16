import numpy as np
import matplotlib.pyplot as plt
import math
import random
import gradDescent
from gradientAllocation import randomParams, stratifiedSampling
from scipy.spatial import ConvexHull
from tqdm import tqdm


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

    # attempting to reverse for minimum rather than maximum
    fSet = [-i for i in fSet]


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
# for finding the minimum
def metaMaxSearch(f, k, d, maxBudget, batchSize, numEvalsPerGrad,
                  a=.02, c=.001, startPos=False, useSPSA=False, useTqdm=False):
    instances = [[] for i in range(k)]
    xHats = [None] * k
    fHats = [None] * k
    numSamples = [0] * k
    # n = [1]*k (why 1 and not 0?)
    n = [1]*k

    if useSPSA:
        gradientDescentObject = gradDescent.SPSA()
    else:
        gradientDescentObject = gradDescent.finiteDifs()
    elapsedBudget = 0

    if not startPos:
        startPositions = stratifiedSampling(d, k)

    else:
        startPositions = startPos

    for i in range(k):
        startPoint = startPositions[i]
        xHats[i] = startPoint
        fHats[i] = f(startPoint)
        elapsedBudget += 1
        numSamples[i] += 1
        instances[i].append((startPoint, fHats[i]))


    # change True to while budget < maxBudget
    convergeDic = {}
    sampleDic = {}

    if useTqdm:
        tqdmTotal = maxBudget - elapsedBudget
        with tqdm(total=tqdmTotal) as pbar:
            while elapsedBudget < maxBudget:
                oldElapsedBudget = elapsedBudget


                hVals = [0]*k
                b = sum(n)
                for i in range(k):
                    hVals[i] = h(n[i], b)
                selectedPoints = selectPoints(hVals, fHats)
                sampleAlloc = allocateSamples(k, selectedPoints, batchSize)
                sampleDic[elapsedBudget] = numSamples.copy()

                # perform sampleAlloc[i] steps for every instance
                # could add in multi-threading here
                for i in range(k):
                    samples = sampleAlloc[i]
                    for j in range(samples):
                        n[i] += 1
                        oldX = instances[i][-1][0]
                        # step from the previous point of the ith instance once
                        partials = gradientDescentObject.partials(f, oldX, numSamples[i], c=c)
                        partials = np.negative(partials)
                        newX = gradientDescentObject.step(oldX, numSamples[i], partials, a=a)
                        instances[i].append((newX, f(newX)))
                        elapsedBudget += numEvalsPerGrad + 1

                        fVal = f(instances[i][-1][0])
                        elapsedBudget += 1

                        if fVal < fHats[i]:
                            fHats[i] = fVal
                            xHats[i] = instances[i][-1]

                        convergeDic[elapsedBudget] = min(fHats)

                        numSamples[i] += numEvalsPerGrad + 2
                # convergeDic[elapsedBudget] = min(fHats)

                pbar.update(elapsedBudget - oldElapsedBudget)

    else:
        while elapsedBudget < maxBudget:

            hVals = [0] * k
            b = sum(n)
            for i in range(k):
                hVals[i] = h(n[i], b)
            selectedPoints = selectPoints(hVals, fHats)
            sampleAlloc = allocateSamples(k, selectedPoints, batchSize)
            sampleDic[elapsedBudget] = numSamples.copy()

            # perform sampleAlloc[i] steps for every instance
            # could add in multi-threading here
            for i in range(k):
                samples = sampleAlloc[i]
                for j in range(samples):
                    n[i] += 1
                    oldX = instances[i][-1][0]
                    # step from the previous point of the ith instance once
                    partials = gradientDescentObject.partials(f, oldX, numSamples[i], c=c)
                    partials = np.negative(partials)
                    newX = gradientDescentObject.step(oldX, numSamples[i], partials, a=a)
                    instances[i].append((newX, f(newX)))
                    elapsedBudget += numEvalsPerGrad + 1

                    fVal = f(instances[i][-1][0])
                    elapsedBudget += 1

                    if fVal < fHats[i]:
                        fHats[i] = fVal
                        xHats[i] = instances[i][-1]

                    convergeDic[elapsedBudget] = min(fHats)

                    numSamples[i] += numEvalsPerGrad + 2
            # convergeDic[elapsedBudget] = min(fHats)

    maxIndex = np.argmax(fHats)
    return (xHats[maxIndex], fHats[maxIndex], convergeDic, instances, numSamples, sampleDic)


def metaMaxInfiniteSearch(f, d, maxBudget, numEvalsPerGrad,
                          a=.02, c=.001, useSPSA=False, useTqdm=False):
    instances = []
    xHats = []
    fHats = []
    numSamples = []
    n = []

    if useSPSA:
        gradientDescentObject = gradDescent.SPSA()
    else:
        gradientDescentObject = gradDescent.finiteDifs()
    elapsedBudget = 0

    convergeDic = {}
    sampleDic = {}

    round = 0
    if useTqdm:
        tqdmTotal = maxBudget - elapsedBudget
        with tqdm(total=tqdmTotal) as pbar:

            while elapsedBudget < maxBudget:
                oldElapsedBudget = elapsedBudget

                # make a new instance
                newX = randomParams(d)
                xHats.append(newX)
                fHats.append(f(newX))
                elapsedBudget += 1
                numSamples.append(1)
                instances.append([(newX, fHats[round]),])
                n.append(1)
                round += 1


                hVals = [0]*round
                b = sum(n)
                for i in range(round):
                    hVals[i] = h(n[i], b)
                selectedPoints = selectPoints(hVals, fHats)
                # there should be no batch size, every selected point should be stepped
                sampleAlloc = allocateSamples(round, selectedPoints, len(selectedPoints))
                sampleDic[elapsedBudget] = numSamples.copy()

                # perform sampleAlloc[i] steps for every instance
                # could add in multi-threading here
                for i in range(round):
                    samples = sampleAlloc[i]
                    for j in range(samples):
                        n[i] += 1
                        oldX = instances[i][-1][0]
                        # step from the previous point of the ith instance once
                        partials = gradientDescentObject.partials(f, oldX, numSamples[i], c=c)
                        partials = np.negative(partials)
                        newX = gradientDescentObject.step(oldX, numSamples[i], partials, a=a)
                        instances[i].append((newX, f(newX)))
                        elapsedBudget += numEvalsPerGrad + 1

                        fVal = f(instances[i][-1][0])
                        elapsedBudget += 1

                        if fVal < fHats[i]:
                            fHats[i] = fVal
                            xHats[i] = instances[i][-1]

                        convergeDic[elapsedBudget] = min(fHats)

                        numSamples[i] += numEvalsPerGrad + 2
                # convergeDic[elapsedBudget] = min(fHats)

                pbar.update(elapsedBudget - oldElapsedBudget)


    else:
        while elapsedBudget < maxBudget:

            # make a new instance
            newX = randomParams(d)
            xHats.append(newX)
            fHats.append(f(newX))
            elapsedBudget += 1
            numSamples.append(1)
            instances.append([(newX, fHats[round]), ])
            n.append(1)
            round += 1

            hVals = [0] * round
            b = sum(n)
            for i in range(round):
                hVals[i] = h(n[i], b)
            selectedPoints = selectPoints(hVals, fHats)
            # there should be no batch size, every selected point should be stepped
            sampleAlloc = allocateSamples(round, selectedPoints, len(selectedPoints))
            sampleDic[elapsedBudget] = numSamples.copy()

            # perform sampleAlloc[i] steps for every instance
            # could add in multi-threading here
            for i in range(round):
                samples = sampleAlloc[i]
                for j in range(samples):
                    n[i] += 1
                    oldX = instances[i][-1][0]
                    # step from the previous point of the ith instance once
                    partials = gradientDescentObject.partials(f, oldX, numSamples[i], c=c)
                    partials = np.negative(partials)
                    newX = gradientDescentObject.step(oldX, numSamples[i], partials, a=a)
                    instances[i].append((newX, f(newX)))
                    elapsedBudget += numEvalsPerGrad + 1

                    prevMin = min(fHats)

                    fVal = f(instances[i][-1][0])
                    elapsedBudget += 1

                    if fVal < fHats[i]:
                        fHats[i] = fVal
                        xHats[i] = instances[i][-1]


                    convergeDic[elapsedBudget] = min(fHats)

                    numSamples[i] += numEvalsPerGrad + 2


    convergeKeys = list(convergeDic.keys())
    convergeKeys.sort()
    newConvergeDic = {}
    for i in range(len(convergeKeys) - 1):

        prevKey = convergeKeys[i]
        prevVal = convergeDic[prevKey]
        nextKey = convergeKeys[i+1]
        nextVal = convergeDic[nextKey]

        newConvergeDic[prevKey] = prevVal

        if nextKey > maxBudget:
            break

        # go through all of the numbers between the two keys
        for j in range(prevKey+1, nextKey):
            # do linear interpolation
            interp = (nextVal-prevVal) * (j-prevKey)/(nextKey-prevKey) + prevVal
            newConvergeDic[j] = interp

    convergeDic = newConvergeDic
    sortedDic = {}
    sortedDic = {}
    sortedKeys = sorted(list(convergeDic.keys()))
    for key in sortedKeys:
        sortedDic[key] = convergeDic[key]

    k = len(numSamples)
    sampleKeys = list(sampleDic.keys())
    for i in range(len(sampleKeys)):
        remainderList = [0]*(k-i-1)
        sampleDic[sampleKeys[i]] += remainderList

    maxIndex = np.argmax(fHats)
    return (xHats[maxIndex], fHats[maxIndex], sortedDic, instances, numSamples, sampleDic)

