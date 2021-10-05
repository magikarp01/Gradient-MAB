import math

import numpy as np


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

    def calcVariance(values):
        numer = 0
        avg = sum(values) / len(values)
        for fVal in values:
            numer += (fVal - avg) ** 2
        return numer / (len(values) + 1)

    def budgetCalc(values, variances, numSamples):
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

        return budget

    # allocate samples given a budget allocation with fractions and a whole number batch size
    # returns array of integers
    def allocateSamples(budgetAlloc, batchSize):
        totSize = sum(budgetAlloc)
        fracParts = {}
        intParts = []
        for i in range(len(budgetAlloc)):
            est = budgetAlloc[i] * batchSize / totSize
            intParts.append(math.floor(est))
            fracParts[est - intParts[i]] = i

        residue = int(round(batchSize - sum(intParts)))
        sortedFracParts = sorted(fracParts.keys(), reverse=True)

        for r in range(residue):
            intParts[fracParts[sortedFracParts[r]]] += 1

        return intParts

    def getBudget(values, variances, numSamples, batchSize, c):
        return OCBA.allocateSamples( OCBA.budgetCalc(values,variances,numSamples), batchSize)


class UCB:
    # problem: c is too small without quadEstMin, all samples go to one thing
    # c should include the upper bound on the value
    def budgetCalc(values, numSamples, c=math.sqrt(2)):
        # reverse since lower is better
        values = [-i for i in values]

        bestIndex = 0

        k = len(numSamples)

        for i in range(k):
            uncertainty = math.log(sum(numSamples)) / numSamples[i]
            uncertainty = math.sqrt(uncertainty)
            uncertainty *= c
            uncertainty += values[i]
            try:
                if uncertainty > maxVal:
                    maxVal = uncertainty
                    bestIndex = i
            except:
                maxVal = uncertainty

        budget = [0] * k
        budget[bestIndex] = 1
        return budget

    # budgetAlloc should just be [0, 1, 0, 0,...]
    def allocateSamples(budgetAlloc, batchSize):
        return [i * batchSize for i in budgetAlloc]

    def getBudget(values, variances, numSamples, batchSize, c):
        pooledVar = sum(variances)/len(variances)
        c *= math.sqrt(pooledVar)
        return UCB.allocateSamples(UCB.budgetCalc(values, numSamples, c), batchSize)


# has functionality for both discount and sliding window
class discountedUCB:

    # used for one specific arm
    # windowLength <= t
    def discountUCBs(valueHistory, discountFactor, windowLength, c):
        denoms = []
        empAverages = []
        for i in range(len(valueHistory)):
            prevValues = valueHistory[i]

            # take reverse since minimum
            prevValues = [-i for i in prevValues]

            t = len(prevValues)
            # if windowLength is default value, consider every previous value
            if windowLength == -1:
                windowLength = t
            # if there are less previous values than window length, consider all of them
            elif windowLength > len(prevValues):
                windowLength = t

            denom = (1 - discountFactor ** windowLength)/(1-discountFactor)
            denoms.append(denom)

            empAverage = 0
            for j in range(windowLength):
                empAverage += prevValues[t-j-1] * (discountFactor ** j)

            empAverage /= denom
            empAverages.append(empAverage)

        totalN = math.log(sum(denoms))
        paddings = []
        k = len(valueHistory)
        for i in range(k):
            paddings.append(c * math.sqrt(totalN / denoms[i]))

        values = [empAverages[i] + paddings[i] for i in range(k)]
        budget = [0] * len(valueHistory)
        budget[np.argmax(values)] = 1
        return budget


    # budgetAlloc should just be [0, 1, 0, 0,...]
    def allocateSamples(budgetAlloc, batchSize):
        return [i * batchSize for i in budgetAlloc]

    def weightedVariance(prevValues, discountFactor, windowLength):
        weightedMean = discountedOCBA.weightedMean(prevValues, discountFactor, windowLength)
        variance = 0

        for i in range(windowLength):
            numer = (prevValues[len(prevValues) - i - 1] - weightedMean)**2
            numer *= (discountFactor ** i)
            variance += numer

        denom = (1 - discountFactor ** windowLength) / (1 - discountFactor)
        variance /= denom

        return variance


    def getBudget(valueHistory, batchSize, c, numSamples, discountFactor, windowLength):
        variances = [discountedUCB.weightedVariance(valueHistory[i], discountFactor, windowLength) for i in
                     range(len(valueHistory))]
        pooledVar = sum(variances)/len(variances)

        # 2 * B from the 2006 UCB discounted paper
        c *= 2 * math.sqrt(pooledVar)

        return discountedUCB.allocateSamples(discountedUCB.discountUCBs(valueHistory, c=c, discountFactor = discountFactor, windowLength = windowLength), batchSize)


class discountedOCBA:
    def weightedMean(prevValues, discountFactor, windowLength):
        denom = (1 - discountFactor ** windowLength)/(1-discountFactor)
        empAverage = 0
        for i in range(windowLength):
            empAverage += prevValues[len(prevValues) - i - 1] * (discountFactor ** i)
        return empAverage/denom

    def weightedVariance(prevValues, discountFactor, windowLength):
        weightedMean = discountedOCBA.weightedMean(prevValues, discountFactor, windowLength)
        variance = 0

        for i in range(windowLength):
            numer = (prevValues[len(prevValues) - i - 1] - weightedMean)**2
            numer *= (discountFactor ** i)
            variance += numer

        denom = (1 - discountFactor ** windowLength) / (1 - discountFactor)
        variance /= denom

        return variance

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

    def budgetCalc(valueHistory, discountFactor, windowLength, numSamples):
        values = [discountedOCBA.weightedMean(valueHistory[i], discountFactor, windowLength) for i in range(len(valueHistory))]
        variances = [discountedOCBA.weightedVariance(valueHistory[i], discountFactor, windowLength) for i in range(len(valueHistory))]

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

        return budget

    # allocate samples given a budget allocation with fractions and a whole number batch size
    # returns array of integers
    def allocateSamples(budgetAlloc, batchSize):
        totSize = sum(budgetAlloc)
        fracParts = {}
        intParts = []
        for i in range(len(budgetAlloc)):
            est = budgetAlloc[i] * batchSize / totSize
            intParts.append(math.floor(est))
            fracParts[est - intParts[i]] = i

        residue = int(round(batchSize - sum(intParts)))
        sortedFracParts = sorted(fracParts.keys(), reverse=True)

        for r in range(residue):
            intParts[fracParts[sortedFracParts[r]]] += 1

        return intParts

    def getBudget(valueHistory, batchSize, c, numSamples, discountFactor, windowLength):
        return discountedOCBA.allocateSamples(
            discountedOCBA.budgetCalc(valueHistory, discountFactor, windowLength, numSamples), batchSize)


