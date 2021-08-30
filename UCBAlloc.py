import numpy as np
import matplotlib.pyplot as plt
import math
import random
import gradDescent
import kriging
import pyDOE
import functions
from gradientAllocation import stratifiedSampling
from tqdm import tqdm


def getBudget(values, numSamples, c=math.sqrt(2)):
    #reverse since lower is better
    values = [-i for i in values]

    bestIndex = 0

    k = len(numSamples)

    for i in range(k):
        uncertainty = math.log(sum(numSamples)) / numSamples[i]
        uncertainty = math.sqrt(uncertainty)
        uncertainty *= c
        uncertainty += values[i]
        try:
            if uncertainty > maxVal:
                maxVal = uncertainty
                bestIndex = i
        except:
            maxVal = uncertainty

    budget = [0]*k
    budget[bestIndex] = 1
    return budget


# budgetAlloc should just be [0, 1, 0, 0,...]
def allocateSamples(budgetAlloc, batchSize):
    return [i*batchSize for i in budgetAlloc]


# minSamples will be done at the start so theres enough points to fit
# assuming we want to minimize a function
# using finite differences for partials, not SPSA
# maxBudget must be much greater than k*minSamples*numEvalsPerGrad, min bound is k*(minSamples*numEvalsPerGrad + 1)
# k is number of instances
def UCBSearch(f, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
              discountRate=.8, a=.001, c=.001, startPos = False, useSPSA=False, useTqdm=False):
    instances = [None]*k
    xHats = [None] * k
    fHats = [None] * k
    estMins = [None] * k
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


                for i in range(k):
                    points = []
                    pointValues = []
                    for point in instances[i]:
                        # each point is (xValue, fValue)
                        points.append(point[0])
                        pointValues.append(point[1])


                    estMins[i], variances[i] = kriging.quadEstMin(points, pointValues, discountRate)

                budgetAlloc = getBudget(estMins, numSamples)
                # sample allocation is the actual allocations to give to each
                sampleAlloc = allocateSamples(budgetAlloc, batchSize)
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
            for i in range(k):
                points = []
                pointValues = []
                for point in instances[i]:
                    # each point is (xValue, fValue)
                    points.append(point[0])
                    pointValues.append(point[1])


                estMins[i], variances[i] = kriging.quadEstMin(points, pointValues, discountRate)

            budgetAlloc = getBudget(estMins, numSamples)
            # sample allocation is the actual allocations to give to each
            sampleAlloc = allocateSamples(budgetAlloc, batchSize)
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


