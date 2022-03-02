import math

import newBaiAllocations
import rewardModels
import metaMaxAlloc
import random

# returns the allocMethod function
# doesn't work I think because the inner function can't be pickled
def baiAllocate(rewardModel, policy):
    def allocMethod(instances, allocSize, discountFactor=.9, slidingWindow=15):
        rewardCalcs = [rewardModel(instance, discountFactor=discountFactor, slidingWindow=slidingWindow) for instance in instances]
        values = [calc[0] for calc in rewardCalcs]
        variances = [calc[1] for calc in rewardCalcs]
        numSamples = [instance.get_numSamples() for instance in instances]

        return policy(values, variances, numSamples, allocSize)

    return allocMethod

# mabPolicies = [newBaiAllocations.OCBA.getBudget, newBaiAllocations.UCB.getBudget]
# models = [rewardModels.restless, rewardModels.trad]


def restlessOCBA(instances, allocSize, discountFactor=.9, slidingWindow=15):
    rewardModel = rewardModels.restless
    policy = newBaiAllocations.OCBA.getBudget
    rewardCalcs = [rewardModel(instance, discountFactor=discountFactor, slidingWindow=slidingWindow) for instance in
                   instances]
    values = [calc[0] for calc in rewardCalcs]
    variances = [calc[1] for calc in rewardCalcs]
    numSamples = [instance.get_numSamples() for instance in instances]
    return policy(values, variances, numSamples, allocSize)

def restlessUCB(instances, allocSize, discountFactor=.9, slidingWindow=15):
    rewardModel = rewardModels.restless
    policy = newBaiAllocations.UCB.getBudget
    rewardCalcs = [rewardModel(instance, discountFactor=discountFactor, slidingWindow=slidingWindow) for instance in
                   instances]
    values = [calc[0] for calc in rewardCalcs]
    variances = [calc[1] for calc in rewardCalcs]
    numSamples = [instance.get_numSamples() for instance in instances]
    return policy(values, variances, numSamples, allocSize)

def tradOCBA(instances, allocSize, discountFactor=.9, slidingWindow=15):
    rewardModel = rewardModels.trad
    policy = newBaiAllocations.OCBA.getBudget
    rewardCalcs = [rewardModel(instance, discountFactor=discountFactor, slidingWindow=slidingWindow) for instance in
                   instances]
    values = [calc[0] for calc in rewardCalcs]
    variances = [calc[1] for calc in rewardCalcs]
    numSamples = [instance.get_numSamples() for instance in instances]
    return policy(values, variances, numSamples, allocSize)

def tradUCB(instances, allocSize, discountFactor=.9, slidingWindow=15):
    rewardModel = rewardModels.trad
    policy = newBaiAllocations.UCB.getBudget
    rewardCalcs = [rewardModel(instance, discountFactor=discountFactor, slidingWindow=slidingWindow) for instance in
                   instances]
    values = [calc[0] for calc in rewardCalcs]
    variances = [calc[1] for calc in rewardCalcs]
    numSamples = [instance.get_numSamples() for instance in instances]
    return policy(values, variances, numSamples, allocSize)




def uniform(instances, allocSize, discountFactor=.9, slidingWindow=15):
    k = len(instances)

    if allocSize < k:
        return random.sample(range(0, k), allocSize)
    else:
        return range(0, k)



def metaMax(instances, allocSize, discountFactor=.9, slidingWindow=15):
    fHats = [instance.get_fHat() for instance in instances]
    nSet = [instance.get_numSamples() for instance in instances]
    t = sum(nSet)

    hSet = [math.exp(-n / math.sqrt(t)) for n in nSet]

    return metaMaxAlloc.selectPoints(hSet, fHats)



