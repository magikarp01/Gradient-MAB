import newBaiAllocations
import rewardModels
import metaMaxAlloc

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


