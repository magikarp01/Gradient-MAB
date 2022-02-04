import math

import newBaiAllocations
import rewardModels
import metaMaxAlloc
import random

# returns the allocMethod function
def baiAllocate(rewardModel, policy):
    def allocMethod(instances, batchSize):
        rewardCalcs = [rewardModel(instance) for instance in instances]
        values = [calc[0] for calc in rewardCalcs]
        variances = [calc[1] for calc in rewardCalcs]
        numSamples = [instance.get_numSamples() for instance in instances]

        return policy(values, variances, numSamples, batchSize)

    return allocMethod


def uniform(instances, batchSize):
    k = len(instances)
    samplesPerInstance = batchSize // k
    results = [samplesPerInstance] * k
    residue = batchSize % k
    extras = random.sample(range(k), residue)
    for i in extras:
        results[i] += 1
    return results


def metaMax(instances, batchSize):
    fHats = [instance.get_fHat() for instance in instances]
    nSet = [instance.get_numSamples() for instance in instances]
    t = sum(nSet)

    hSet = [math.exp(-n / math.sqrt(t)) for n in nSet]

    selectedPoints = metaMaxAlloc.selectPoints(hSet, fHats)



