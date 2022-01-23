import numpy as np
import math
import gradDescent
from gradientAllocation import stratifiedSampling
from tqdm import tqdm
from instance import Instance
from multiprocessing import Process, Queue

# rewardModel is fit, restless, trad
# baiBudget is getBudget function from OCBA or UCB
# minSamples will be done at the start so theres enough points to fit
# assuming we want to minimize a function
# using finite differences for partials, not SPSA
# maxBudget must be much greater than k*minSamples*numEvalsPerGrad, min bound is k*(minSamples*numEvalsPerGrad + 1)
# k is number of instances
# allocMethod is one of the getBudget methods from baiAllocations
def MABSearch(rewardModel, baiBudget, f, k, d, maxBudget, numProcesses,
              numEvalsPerGrad, minSamples,
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

        rewardCalcs = [rewardModel(instance) for instance in instances]

        values = [calc[0] for calc in rewardCalcs]
        variances = [calc[1] for calc in rewardCalcs]
        numSamples = [instance.get_numSamples() for instance in instances]

        sampleDic[elapsedBudget] = numSamples.copy()

        sampleAlloc = baiBudget(values, variances, numSamples, numProcesses)

        # perform sampleAlloc[i] steps for every instance
        # could add in multi-threading here

        # it's possible that instance.descend has to be changed to be descend(instance), returns new instance
        # Process may be messed up
        processes = [Process(target=instances[x].descend, args=()) for x in range(len(sampleAlloc))]

        for p in processes:
            p.start()

        for p in processes:
            # not sure if this should go here or with p.start()
            elapsedBudget += numEvalsPerGrad + 2
            convergeDic[elapsedBudget] = min([instance.get_fHat()] for instance in instances)
            p.join()



        # convergeDic[elapsedBudget] = min(fHats)

        if(useTqdm):
            pbar.update(elapsedBudget - oldElapsedBudget)

    maxIndex = np.argmax([instance.get_fHat() for instance in instances])
    return (instances[maxIndex].get_xHat(), instances[maxIndex].get_fHat(),
            convergeDic, instances, [instance.get_numSamples() for instance in instances], sampleDic)
