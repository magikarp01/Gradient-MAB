import math

import numpy as np

import kriging

# allocates one to best allocSize arms

class OCBA:
    # takes negative differences
    def getKroneckers(values):
        numInstances = len(values)

        minValue = values[0]
        for i in range(numInstances):
            if values[i] < minValue:
                minValue = values[i]

        kroneckers = [0] * numInstances
        for i in range(numInstances):
            # negative of usual, but since we're doing minimums it should work
            kroneckers[i] = values[i] - minValue

        return kroneckers

    def starvingCalc(values, variances, numSamples):
        kroneckers = OCBA.getKroneckers(values)
        numInstances = len(variances)

        optimalInstances = []
        for minIndex in range(numInstances):
            if kroneckers[minIndex] == 0:
                optimalInstances.append(minIndex)

        for instanceIndex in range(numInstances):
            if instanceIndex not in optimalInstances:
                if variances[instanceIndex] != 0 and kroneckers[instanceIndex] != 0:
                    startInstance = instanceIndex
                    break
        try:
            denom = variances[startInstance] / (kroneckers[startInstance] ** 2)
        except:  # happens when all of the nonoptimal moves have zero variance
            varFlag = False
        ratios = [0] * numInstances

        for instanceIndex in range(numInstances):
            # try:
            #     print(startInstance)
            # except:
            #     print("problem")
            if instanceIndex in optimalInstances:
                continue
            elif instanceIndex == startInstance:
                ratios[instanceIndex] = 1
                continue
            else:
                ratio = variances[instanceIndex] / (kroneckers[instanceIndex] ** 2)
                try:
                    ratio /= denom
                except:  # happens when all of the nonoptimal moves have zero variance, only one possible move
                    budget = [0] * numInstances
                    totBudget = 0
                    for action in range(numInstances):
                        totBudget += numSamples[action]
                    averageAlloc = (totBudget + 1) / len(optimalInstances)
                    for x in optimalInstances:
                        budget[x] = averageAlloc

                    return budget

                ratios[instanceIndex] = ratio

        # set up calculation for the proportion of the budget for optimal actions
        tempSum = 0  # sum of N^2/stdev
        for action in range(numInstances):
            if ratios[action] != 0:
                tempSum += ratios[action] ** 2 / variances[action]  # stuff cancels
        tempSum = math.sqrt(tempSum)

        for action in optimalInstances:  # not sure if optimalProportion might be different for multiple optimal actions
            ratios[action] = math.sqrt(variances[action]) * tempSum

        # calculate the actual budgets from the proportions
        total = 0
        budget = [0] * numInstances
        totBudget = 0
        for action in range(numInstances):
            total += ratios[action]
            totBudget += numSamples[action]

        try:
            scale = (totBudget + 1) / total
            for action in range(numInstances):
                budget[action] = scale * ratios[action]
        except:
            uniform = (totBudget + 1) / numInstances
            for action in range(numInstances):
                budget[action] = uniform

        starving = [budget[i] - numSamples[i] for i in range(numInstances)]
        return starving



    # def getBudget(values, variances, numSamples, allocSize, c):
    def getBudget(values, variances, numSamples, allocSize, c=math.sqrt(2)):
        starving = OCBA.starvingCalc(values,variances,numSamples)
        starvingSorted = sorted(range(len(starving)), key= lambda x : starving[x], reverse=True)

        # should make sure there are more instances than batch size
        if allocSize > len(starvingSorted):
            return starvingSorted
        else:
            return starvingSorted[:allocSize]



class UCB:
    # problem: c is too small without quadEstMin, all samples go to one thing
    # c should include the upper bound on the value
    def budgetCalc(values, numSamples, c=math.sqrt(2)):
        # reverse since lower is better
        values = [-i for i in values]

        bestIndex = 0

        k = len(numSamples)

        uncertainties = []

        for i in range(k):
            uncertainty = math.log(sum(numSamples)) / numSamples[i]
            uncertainty = math.sqrt(uncertainty)
            uncertainty *= c
            uncertainty += values[i]

            uncertainties.append(uncertainty)

        return uncertainties


    # def getBudget(values, variances, numSamples, allocSize, c):
    def getBudget(values, variances, numSamples, allocSize, c=math.sqrt(2)):
        pooledVar = sum(variances)/len(variances)
        newC = c * math.sqrt(pooledVar)
        uncertainties = UCB.budgetCalc(values, numSamples, c=newC)

        uncertaintiesSorted = sorted(range(len(uncertainties)), key=lambda x: uncertainties[x], reverse=True)

        # should make sure there are more instances than batch size
        if allocSize > len(uncertaintiesSorted):
            return uncertaintiesSorted
        else:
            return uncertaintiesSorted[:allocSize]