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
    variances = [None] * k
    numSamples = [0] * k

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

        # this is for minimizing not maximizing
        instances[i] = gradientDescentObject.gradDescent(f, startPoint, minSamples, a, c)[2]
        elapsedBudget += minSamples * numEvalsPerGrad
        numSamples[i] += minSamples * numEvalsPerGrad

    # change True to while budget < maxBudget
    convergeDic = {}
    sampleDic = {}

    if useTqdm:
        tqdmTotal = maxBudget-elapsedBudget
        with tqdm(total=tqdmTotal) as pbar:
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

                pbar.update(elapsedBudget - oldElapsedBudget)

    else:
        while elapsedBudget < maxBudget:
            # print(elapsedBudget)

            valueHistory = []

            for i in range(k):
                improvements = []
                for t in range(len(instances[i]) - 1):
                    improvement = instances[i][t + 1][1] - instances[i][t][1]
                    # improvement depends on step size: divide by step size
                    improvement /= gradientDescentObject.get_ct(c, t)

                    if improvement < 0:
                        improvement = 0
                    improvements.append(improvement)

                valueHistory.append(improvement)

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

    maxIndex = np.argmax(fHats)
    return (xHats[maxIndex], fHats[maxIndex], convergeDic, instances, numSamples, sampleDic)

