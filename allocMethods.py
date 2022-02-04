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

    if batchSize < k:
        return random.sample(range(0, k), batchSize)
    else:
        return range(0, k)



def metaMax(instances, batchSize):
    fHats = [instance.get_fHat() for instance in instances]
    nSet = [instance.get_numSamples() for instance in instances]
    t = sum(nSet)

    hSet = [math.exp(-n / math.sqrt(t)) for n in nSet]

    return metaMaxAlloc.selectPoints(hSet, fHats)



