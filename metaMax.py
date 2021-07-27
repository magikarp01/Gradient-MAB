import numpy as np
import matplotlib
import math
import random
from scipy.spatial import ConvexHull
import gradDescent

SPSAObject = gradDescent.SPSA()
finiteDifsObject = gradDescent.finiteDifs()
from tqdm import tqdm

# all params are in range [0, 1)
def randomParams(d):
    vec = [None]*d
    for i in range(d):
        vec[i] = random.random()
    return np.array(vec)


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


# f is loss function to be evaluated
# k is number of instances to run
# d is num of dimensions
# budget is number that can be allocated
# this metaMax gets the maximum, not the minimum
def SPSANumRounds(f, k, d, numRounds):

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
            partials = np.negative(SPSAObject.partials(f, xPos[pointIndex], n[pointIndex]))
            xPos[pointIndex] = SPSAObject.step(xPos[pointIndex], n[pointIndex], partials)
            n[pointIndex] += 1
            fVal = f(xPos[pointIndex])
            if fVal > fHats[pointIndex]:
                fHats[pointIndex] = fVal
                xHats[pointIndex] = xPos[pointIndex]
            # possibly budget += 1

        budget += 3*len(selected)

    maxIndex = np.argmax(fHats)
    return (xHats[maxIndex], fHats[maxIndex], n, budget)


def SPSABudget(f, k, d, maxBudget):
    n = [1] * k
    xPos = [None] * k  # current position
    xHats = [None] * k  # argmax over xPos so far
    fHats = [None] * k  # max values of f, corresponding to xHats

    for i in range(k):
        startPoint = randomParams(d)
        xPos[i] = startPoint
        xHats[i] = startPoint
        fHats[i] = f(startPoint)

    # budget = t_r, function has already been evaluated k times
    budget = k

    round = -1

    convergeDic = {}
    while budget < maxBudget:
        round += 1
        # default is that the instance was not sampled

        # find the hVals for this round
        hVals = [None] * k
        for i in range(k):
            hVals[i] = h(n[i], budget)

        # select the points for sampling
        selected = selectPoints(hVals, fHats)

        # update the selected points

        for pointIndex in selected:
            partials = np.negative(SPSAObject.partials(f, xPos[pointIndex], n[pointIndex]))
            oldXPos = xPos[pointIndex]
            xPos[pointIndex] = SPSAObject.step(xPos[pointIndex], n[pointIndex], partials)
            n[pointIndex] += 1
            fVal = f(xPos[pointIndex])
            if fVal > fHats[pointIndex]:
                fHats[pointIndex] = fVal
                xHats[pointIndex] = xPos[pointIndex]
            else:
                xPos[pointIndex] = oldXPos
            # possibly budget += 1

        budget += 3 * len(selected)
        convergeDic[budget] = max(fHats)

    maxIndex = np.argmax(fHats)
    return (xHats[maxIndex], fHats[maxIndex], convergeDic)


def SPSABudgetGivenX(f, k, d, maxBudget, xPos):
    n = [1] * k
    xHats = [None] * k  # argmax over xPos so far
    fHats = [None] * k  # max values of f, corresponding to xHats

    for i in range(k):
        startPoint = randomParams(d)
        xHats[i] = startPoint
        fHats[i] = f(startPoint)

    # budget = t_r, function has already been evaluated k times
    budget = k

    round = -1

    convergeDic = {}
    while budget < maxBudget:
        round += 1
        # default is that the instance was not sampled

        # find the hVals for this round
        hVals = [None] * k
        for i in range(k):
            hVals[i] = h(n[i], budget)

        # select the points for sampling
        selected = selectPoints(hVals, fHats)

        # update the selected points

        for pointIndex in selected:
            partials = np.negative(SPSAObject.partials(f, xPos[pointIndex], n[pointIndex]))
            xPos[pointIndex] = SPSAObject.step(xPos[pointIndex], n[pointIndex], partials)
            n[pointIndex] += 1
            fVal = f(xPos[pointIndex])
            if fVal > fHats[pointIndex]:
                fHats[pointIndex] = fVal
                xHats[pointIndex] = xPos[pointIndex]
            # possibly budget += 1

        budget += 3 * len(selected)
        convergeDic[budget] = max(fHats)

    maxIndex = np.argmax(fHats)
    return (xHats[maxIndex], fHats[maxIndex], convergeDic)


def finiteDifsBudget(f, k, d, maxBudget, a=.16):
    n = [1] * k
    xPos = [None] * k  # current position
    xHats = [None] * k  # argmax over xPos so far
    fHats = [None] * k  # max values of f, corresponding to xHats

    for i in range(k):
        startPoint = randomParams(d)
        xPos[i] = startPoint
        xHats[i] = startPoint
        fHats[i] = f(startPoint)

    # budget = t_r, function has already been evaluated k times
    budget = k

    round = -1

    convergeDic = {}
    while budget < maxBudget:
        round += 1
        # default is that the instance was not sampled

        # find the hVals for this round
        hVals = [None] * k
        for i in range(k):
            hVals[i] = h(n[i], budget)

        # select the points for sampling
        selected = selectPoints(hVals, fHats)

        # update the selected points

        for pointIndex in selected:
            partials = np.negative(finiteDifsObject.partials(f, xPos[pointIndex], n[pointIndex]))
            oldXPos = xPos[pointIndex]

            # should step in the positive direction (dir. of max ascent)
            xPos[pointIndex] = finiteDifsObject.step(xPos[pointIndex], n[pointIndex], partials, a)
            n[pointIndex] += 1
            fVal = f(xPos[pointIndex])
            if fVal > fHats[pointIndex]:
                fHats[pointIndex] = fVal
                xHats[pointIndex] = xPos[pointIndex]
            else:
                xPos[pointIndex] = oldXPos
            # possibly budget += 1

        budget += 3 * len(selected)
        convergeDic[budget] = max(fHats)

    maxIndex = np.argmax(fHats)
    return (xHats[maxIndex], fHats[maxIndex], hVals, convergeDic)