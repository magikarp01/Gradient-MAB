import numpy as np
import math
import gradDescent
import baiAllocations
from gradientAllocation import stratifiedSampling, randomParams
from tqdm import tqdm


# minSamples will be done at the start so theres enough points to fit
# assuming we want to minimize a function
# using finite differences for partials, not SPSA
# maxBudget must be much greater than k*minSamples*numEvalsPerGrad, min bound is k*(minSamples*numEvalsPerGrad + 1)
# k is number of instances
# allocMethod is one of the getBudget methods from baiAllocations
def MABSearch(allocMethod, f, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                  a=.001, c=.001, startPos = False, useSPSA=False, useTqdm=False, UCBPad = math.sqrt(2)):

    instances = []

    if useSPSA:
        gradientDescentObject = gradDescent.SPSA(a=a, c=c)
    else:
        gradientDescentObject = gradDescent.finiteDifs(a=a, c=c)

    if not startPos:
        startPositions = stratifiedSampling(d, k)

    else:
        startPositions = startPos

    for i in range(k):
        startPoint = startPositions[i]
        xHats[i] = startPoint
        fHats[i] = f(startPoint)
        # elapsedBudget += 1
        numSamples[i] += 1

        # this is for minimizing not maximizing
        instances[i] = gradientDescentObject.gradDescent(f, startPoint, minSamples, a, c)[2]
        # elapsedBudget += minSamples * numEvalsPerGrad
        numSamples[i] += minSamples * numEvalsPerGrad

    # change True to while budget < maxBudget
    convergeDic = {}
    sampleDic = {}
    elapsedBudget = 0

    if useTqdm:
        tqdmTotal = maxBudget - elapsedBudget
        pbar = tqdm(total=tqdmTotal)
    while elapsedBudget < maxBudget:
        oldElapsedBudget = elapsedBudget
        # print(elapsedBudget)

        for i in range(k):
            points = []
            pointValues = []
            for point in instances[i]:
                # each point is (xValue, fValue)
                points.append(point[0])
                pointValues.append(point[1])

            variances[i] = baiAllocations.OCBA.calcVariance([pt[1] for pt in instances[i]])


        # sample allocation is the actual allocations to give to each
        sampleAlloc = allocMethod(fHats, variances, numSamples, batchSize, UCBPad)
        sampleDic[elapsedBudget] = numSamples.copy()

        # perform sampleAlloc[i] steps for every instance
        # could add in multi-threading here
        for i in range(k):
            samples = sampleAlloc[i]
            for j in range(samples):
                # step from the previous point of the ith instance once

                partials = gradientDescentObject.partials(f, instances[i][-1][0], numSamples[i], c=c)
                partials = np.negative(partials)
                newX = gradientDescentObject.step(instances[i][-1][0], numSamples[i], partials, a=a)
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

        if(useTqdm):
            pbar.update(elapsedBudget - oldElapsedBudget)

    maxIndex = np.argmax(fHats)
    return (xHats[maxIndex], fHats[maxIndex], convergeDic, instances, numSamples, sampleDic)


def tradInfiniteSearch(allocMethod, f, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
            a=.001, c=.001, useSPSA=False, useTqdm=False, UCBPad = math.sqrt(2)):
    instances = []
    xHats = []
    fHats = []
    variances = []
    numSamples = []

    if useSPSA:
        gradientDescentObject = gradDescent.SPSA()
    else:
        gradientDescentObject = gradDescent.finiteDifs()

    elapsedBudget = 0

    # change True to while budget < maxBudget
    convergeDic = {}
    sampleDic = {}

    round = 0
    if useTqdm:
        tqdmTotal = maxBudget-elapsedBudget
        pbar = tqdm(total=tqdmTotal)
    while elapsedBudget < maxBudget:
        oldElapsedBudget = elapsedBudget
        # print(elapsedBudget)

        # make a new instance
        newX = randomParams(d)
        xHats.append(newX)
        fHats.append(f(newX))

        instances.append(gradientDescentObject.gradDescent(f, newX, minSamples, a, c)[2])
        elapsedBudget += 1 + minSamples*numEvalsPerGrad
        numSamples.append(1 + minSamples*numEvalsPerGrad)

        variances.append(None)
        round += 1

        for i in range(round):
            points = []
            pointValues = []
            for point in instances[i]:
                # each point is (xValue, fValue)
                points.append(point[0])
                pointValues.append(point[1])

            variances[i] = baiAllocations.OCBA.calcVariance([pt[1] for pt in instances[i]])

        # sample allocation is the actual allocations to give to each
        sampleAlloc = allocMethod(fHats, variances, numSamples, batchSize, UCBPad)
        sampleDic[elapsedBudget] = numSamples.copy()

        # perform sampleAlloc[i] steps for every instance
        # could add in multi-threading here
        for i in range(round):
            samples = sampleAlloc[i]
            for j in range(samples):
                # step from the previous point of the ith instance once

                partials = gradientDescentObject.partials(f, instances[i][-1][0], numSamples[i], c=c)
                partials = np.negative(partials)
                newX = gradientDescentObject.step(instances[i][-1][0], numSamples[i], partials, a=a)
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

        if useTqdm:
            pbar.update(elapsedBudget - oldElapsedBudget)


    # fix sampleDic to have each list be the same length, unused are 0s
    k = len(numSamples)
    sampleKeys = list(sampleDic.keys())
    for i in range(len(sampleKeys)):
        remainderList = [0]*(k-i-1)
        sampleDic[sampleKeys[i]] += remainderList

    maxIndex = np.argmax(fHats)
    return (xHats[maxIndex], fHats[maxIndex], convergeDic, instances, numSamples, sampleDic)
