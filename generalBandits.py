import numpy as np
import math
import gradDescent
from gradientAllocation import stratifiedSampling
from tqdm import tqdm
from instance import Instance


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
        instances[i] = Instance(f, d, gradientDescentObject, startPositions[i])

    # change True to while budget < maxBudget
    convergeDic = {}
    sampleDic = {}
    elapsedBudget = 0

    if useTqdm:
        tqdmTotal = maxBudget - elapsedBudget
        pbar = tqdm(total=tqdmTotal)
    while elapsedBudget < maxBudget:
        oldElapsedBudget = elapsedBudget
        sampleAlloc = allocMethod(instances, )
        sampleDic[elapsedBudget] = [instance.get_numSamples() for instance in instances]

        # perform sampleAlloc[i] steps for every instance
        # could add in multi-threading here
        for i in range(k):
            samples = sampleAlloc[i]
            for j in range(samples):
                instances[i].descend()
                elapsedBudget += numEvalsPerGrad + 2

                convergeDic[elapsedBudget] = min([instance.get_fHat()] for instance in instances)

        # convergeDic[elapsedBudget] = min(fHats)

        if(useTqdm):
            pbar.update(elapsedBudget - oldElapsedBudget)

    maxIndex = np.argmax([instance.get_fHat() for instance in instances])
    return (instances[maxIndex].get_xHat(), instances[maxIndex].get_fHat(),
            convergeDic, instances, [instance.get_numSamples() for instance in instances], sampleDic)
