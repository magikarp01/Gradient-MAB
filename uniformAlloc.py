import numpy as np
import matplotlib.pyplot as plt
import math
import random
import gradDescent
from gradientAllocation import stratifiedSampling


# batchSize should ideally be multiple of k
def allocateSamples(k, batchSize):
    samplesPerInstance = batchSize // k
    residue = samplesPerInstance % k
    return [samplesPerInstance] * k


# minSamples will be done at the start so theres enough points to fit
# assuming we want to minimize a function
# using finite differences for partials, not SPSA
# maxBudget must be much greater than k*minSamples*numEvalsPerGrad, min bound is k*(minSamples*numEvalsPerGrad + 1)
# maxBudget is total number of function evaluations
# k is number of instances
def uniformSearch(f, k, d, maxBudget, batchSize, numEvalsPerGrad, a=.02, c=.001):
    instances = [[]] * k
    xHats = [None] * k
    fHats = [None] * k
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
        instances[i].append((startPoint, fHats[i]))

    # change True to while budget < maxBudget
    convergeDic = {}
    sampleDic = {}
    while elapsedBudget < maxBudget:

        sampleAlloc = allocateSamples(k, batchSize)

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


