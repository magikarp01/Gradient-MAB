import numpy as np
import matplotlib.pyplot as plt
import math
import random
import gradDescent
import kriging
import pyDOE
import functions
from gradientAllocation import randomParams, stratifiedSampling
from tqdm import tqdm


# UCBPad should depend on maximum bound of reward function, currently way too big
# should ask Dr. Fu what UCBPad would be good

def restlessSearch(allocMethod, discountFactor, windowLength, f, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
            a=.001, c=.001, startPos = False, useSPSA=False, useTqdm=False, UCBPad=math.sqrt(2)):
    instances = [None]*k
    xHats = [None] * k
    fHats = [None] * k
    numSamples = [0] * k

    if useSPSA:
        gradientDescentObject = gradDescent.SPSA()
    else:
        gradientDescentObject = gradDescent.finiteDifs()


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
        tqdmTotal = maxBudget-elapsedBudget
        pbar = tqdm(total=tqdmTotal)
    while elapsedBudget < maxBudget:
        oldElapsedBudget = elapsedBudget
        # print(elapsedBudget)

        valueHistory = []

        for i in range(k):
            improvements = []
            for t in range(len(instances[i])-1):
                improvement = instances[i][t+1][1] - instances[i][t][1]
                # improvement depends on step size: divide by step size
                # improvement should be negative, since in baiAllocations negative values = better
                improvement /= gradientDescentObject.get_ct(c, t)

                if improvement > 0:
                    improvement = 0
                improvements.append(improvement)

            valueHistory.append(improvements)

        # sample allocation is the actual allocations to give to each
        sampleAlloc = allocMethod(valueHistory, batchSize, UCBPad, numSamples, discountFactor, windowLength)
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

        if useTqdm:
            pbar.update(elapsedBudget - oldElapsedBudget)


    maxIndex = np.argmax(fHats)
    return (xHats[maxIndex], fHats[maxIndex], convergeDic, instances, numSamples, sampleDic)


def restlessInfiniteSearch(allocMethod, discountFactor, windowLength, f, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
            a=.001, c=.001, useSPSA=False, useTqdm=False, UCBPad = math.sqrt(2)):
    instances = []
    xHats = []
    fHats = []
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

        round += 1

        valueHistory = []

        for i in range(round):
            improvements = []
            for t in range(len(instances[i])-1):
                improvement = instances[i][t+1][1] - instances[i][t][1]
                # improvement depends on step size: divide by step size
                # improvement should be negative, since in baiAllocations negative values = better
                improvement /= gradientDescentObject.get_ct(c, t)

                if improvement > 0:
                    improvement = 0
                improvements.append(improvement)

            valueHistory.append(improvements)

        # sample allocation is the actual allocations to give to each
        sampleAlloc = allocMethod(valueHistory, batchSize, UCBPad, numSamples, discountFactor, windowLength)
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


