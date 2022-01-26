import numpy as np
import math
import gradDescent
import gradDescent

from gradientAllocation import stratifiedSampling
from tqdm import tqdm
from instance import Instance
from multiprocessing import Process, Queue
# from multiprocessing.managers import SharedMemoryManager
# from multiprocessing.shared_memory import SharedMemory

# rewardModel is fit, restless, trad
# baiBudget is getBudget function from OCBA or UCB
# minSamples will be done at the start so theres enough points to fit
# assuming we want to minimize a function
# using finite differences for partials, not SPSA
# maxBudget must be much greater than k*minSamples*numEvalsPerGrad, min bound is k*(minSamples*numEvalsPerGrad + 1)
# k is number of instances
# allocMethod is one of the getBudget methods from baiAllocations
# numProcesses is number of processes, batchSize is number of gradients for each process
def MABSearch(rewardModel, baiBudget, f, k, d, maxBudget, numEvalsPerGrad, minSamples, numProcesses, batchSize,
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


        sampleAlloc = baiBudget(values, variances, numSamples, numProcesses)

        # Process may be messed up
        # processes = [Process(target=instances[x].descend, args=()) for x in range(len(sampleAlloc))]
        qout = Queue()
        processes = [Process(target=gradDescent.methods.descend,
                             args=(instances[x].get_lastPoint(), f, instances[x].get_numSamples(), a, c, batchSize, qout))
                     for x in range(numProcesses)]

        for p in processes:
            p.start()

        instanceNum = 0
        for p in processes:

            # not sure if this should go here or with p.start()
            p.join()

            newPoints = qout.get()
            instances[instanceNum].multiprocessDescend(newPoints)

            elapsedBudget += (numEvalsPerGrad + 1) * len(newPoints)
            convergeDic[elapsedBudget] = min([instance.get_fHat()] for instance in instances)
            instanceNum += 1


        if(useTqdm):
            pbar.update(elapsedBudget - oldElapsedBudget)

    maxIndex = np.argmax([instance.get_fHat() for instance in instances])
    return (instances[maxIndex].get_xHat(), instances[maxIndex].get_fHat(),
            convergeDic, instances, [instance.get_numSamples() for instance in instances], sampleDic)
    # return instances, convergeDic, sampleDic

