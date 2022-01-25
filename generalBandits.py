import numpy as np
import math
import gradDescent
from gradientAllocation import stratifiedSampling
from tqdm import tqdm
from instance import Instance


# rewardModel is fit, restless, trad
# baiBudget is getBudget function from OCBA or UCB
# minSamples will be done at the start so theres enough points to fit
# assuming we want to minimize a function
# using finite differences for partials, not SPSA
# maxBudget must be much greater than k*minSamples*numEvalsPerGrad, min bound is k*(minSamples*numEvalsPerGrad + 1)
# k is number of instances
# allocMethod is one of the getBudget methods from baiAllocations
def MABSearch(rewardModel, baiBudget, f, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                  a=.001, c=.001, startPos = False, useSPSA=False, useTqdm=False):

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
        newInstance = Instance(f, d, gradientDescentObject, startPositions[i])
        instances.append(newInstance)
        for j in range(minSamples):
            instances[i].descend()


    convergeDic = {}
    sampleDic = {}
    elapsedBudget = 0

    if useTqdm:
        tqdmTotal = maxBudget - elapsedBudget
        pbar = tqdm(total=tqdmTotal, position = 0, leave=True)


    while elapsedBudget < maxBudget:
        oldElapsedBudget = elapsedBudget

        rewardCalcs = [rewardModel(instance) for instance in instances]

        values = [calc[0] for calc in rewardCalcs]
        variances = [calc[1] for calc in rewardCalcs]
        numSamples = [instance.get_numSamples() for instance in instances]

        sampleDic[elapsedBudget] = numSamples.copy()


        sampleAlloc = baiBudget(values, variances, numSamples, batchSize)

        for i in sampleAlloc:
            instances[i].descend()
            elapsedBudget += numEvalsPerGrad + 2

            convergeDic[elapsedBudget] = min([instance.get_fHat()] for instance in instances)

        if(useTqdm):
            pbar.update(elapsedBudget - oldElapsedBudget)

    maxIndex = np.argmax([instance.get_fHat() for instance in instances])
    return (instances[maxIndex].get_xHat(), instances[maxIndex].get_fHat(),
            convergeDic, instances, [instance.get_numSamples() for instance in instances], sampleDic)
    # return instances, convergeDic, sampleDic