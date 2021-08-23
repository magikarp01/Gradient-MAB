import numpy as np
import matplotlib.pyplot as plt
import math
import random
import gradDescent
from gradientAllocation import stratifiedSampling


# batchSize should ideally be multiple of k
def allocateSamples(k, batchSize):
    samplesPerInstance = batchSize // k
    results = [samplesPerInstance] * k
    residue = samplesPerInstance % k
    extras = random.sample(range(k), residue)
    for i in extras:
        results[i] += 1
    return results


# minSamples will be done at the start so theres enough points to fit
# assuming we want to minimize a function
# using finite differences for partials, not SPSA
# maxBudget must be much greater than k*minSamples*numEvalsPerGrad, min bound is k*(minSamples*numEvalsPerGrad + 1)
# maxBudget is total number of function evaluations
# k is number of instances
def thrAscSearch(f, k, d, maxBudget, batchSize, numEvalsPerGrad,
                 a=.02, c=.001, startPos = False):
    instances = [[] for i in range(k)]
    xHats = [None] * k
    fHats = [None] * k
    numSamples = [0] * k

    finiteDifsObject = gradDescent.finiteDifs()
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
    while elapsedBudget < maxBudget:
        # print(elapsedBudget)

        sampleAlloc = allocateSamples(k, batchSize)
        sampleDic[elapsedBudget] = numSamples.copy()

        # perform sampleAlloc[i] steps for every instance
        # could add in multi-threading here
        for i in range(k):
            samples = sampleAlloc[i]
            for j in range(samples):
                # step from the previous point of the ith instance once

                oldX = instances[i][-1][0]
                partials = finiteDifsObject.partials(f, oldX, numSamples[i], c=c)
                partials = np.negative(partials)

                newX = finiteDifsObject.step(oldX, numSamples[i], partials, a=a)
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

