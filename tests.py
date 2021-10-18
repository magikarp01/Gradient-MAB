import unittest
import random

import kriging
import gradDescent
import functions
import matplotlib.pyplot as plt
import gradientAllocation

import uniformAlloc
import metaMaxAlloc

import fitBandits
import restlessBandits
import tradBandits
import baiAllocations

# class TestClass(unittest.TestCase):
class visualize:

    def display1DResults(results, fun, colors, fColor='b', lineWidth=3):
        instances = results[3]
        sampleDic = results[5]
        convergeDic = results[2]
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        plt.subplots_adjust(hspace=.5)
        gradientAllocation.displayInstances1D(fun, instances, ax1, colors, fColor=fColor, lineWidth=lineWidth)
        gradientAllocation.displaySamplingHistory(sampleDic, ax2, colors)
        gradientAllocation.displayMinimaHistory(convergeDic, ax3)

    def display3DResults(results, fun, colors, fColor='b', lineWidth=3, alpha=.1, showFunction=True,
                              fig=plt.figure()):
        instances = results[3]
        sampleDic = results[5]
        convergeDic = results[2]
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(224)
        # fig, (ax1, ax2, ax3) = plt.subplots(3)
        plt.subplots_adjust(hspace=.5)
        gradientAllocation.displayInstances3D(fun, [0, 1], instances, ax1, colors, fColor=fColor, lineWidth=lineWidth,
                                              alpha=alpha, showFunction=showFunction)
        gradientAllocation.displaySamplingHistory(sampleDic, ax2, colors)
        gradientAllocation.displayMinimaHistory(convergeDic, ax3)

    def test_displayNDResults(results, colors, fig=plt.figure()):
        instances = results[3]
        sampleDic = results[5]
        convergeDic = results[2]
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(224)
        # fig, (ax1, ax2, ax3) = plt.subplots(3)
        plt.subplots_adjust(hspace=.5)
        gradientAllocation.displayInstancesND(instances, ax1, colors)
        gradientAllocation.displaySamplingHistory(sampleDic, ax2, colors)
        gradientAllocation.displayMinimaHistory(convergeDic, ax3)


class fitTests:

    # params is [f, k, d, maxBudget, batchSize, numEvalsPerGrad]
    def fitOCBA(sharedParams, minSamples,
                           discountRate=.9, a=.001, c=.001, startPos=False, useTqdm=True, useSPSA=False):
        f = sharedParams[0]
        k = sharedParams[1]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = fitBandits.fitSearch(baiAllocations.OCBA.getBudget, f, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples, discountRate=discountRate,
                                          a=a, c=c, startPos=startPos, useTqdm=useTqdm, useSPSA=useSPSA)
        # return (xHats[maxIndex], fHats[maxIndex], convergeDic, instances, numSamples, sampleDic)
        # print("Convergence Dictionary: ", end="")
        # print(results[2])
        # print()
        #
        # instances = results[3]
        # for i in range(k):
        #     print(f"Instance #{i}: ", end="")
        #     print(instances[i])
        #     print()
        # print("Sample Allocation: ", end="")
        # print(results[4])
        return results

    def fitInfiniteOCBA(sharedParams, minSamples,
                           discountRate=.9, a=.001, c=.001, startPos=False, useTqdm=True, useSPSA=False):
        # UCBSearch(f, k, d, maxBudget, minSamples, batchSize, numEvalsPerGrad)
        f = sharedParams[0]
        k = sharedParams[1]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = fitBandits.fitInfiniteSearch(baiAllocations.OCBA.getBudget, f, d, maxBudget, batchSize, numEvalsPerGrad, minSamples, discountRate=discountRate,
                                          a=a, c=c, useTqdm=useTqdm, useSPSA=useSPSA)
        return results

    def fitUCB(sharedParams, minSamples,
                     discountRate=.9, a=.001, c=.001, startPos=False, useTqdm=True, useSPSA=False):
        # UCBSearch(f, k, d, maxBudget, minSamples, batchSize, numEvalsPerGrad)
        f = sharedParams[0]
        k = sharedParams[1]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = fitBandits.fitSearch(baiAllocations.UCB.getBudget, f, k, d, maxBudget, batchSize, numEvalsPerGrad,
                                       minSamples, discountRate=discountRate,
                                       a=a, c=c, startPos=startPos, useTqdm=useTqdm, useSPSA=useSPSA)
        return results

    def fitInfiniteUCB(sharedParams, minSamples,
                           discountRate=.9, a=.001, c=.001, startPos=False, useTqdm=True, useSPSA=False):
        # UCBSearch(f, k, d, maxBudget, minSamples, batchSize, numEvalsPerGrad)
        f = sharedParams[0]
        k = sharedParams[1]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = fitBandits.fitInfiniteSearch(baiAllocations.UCB.getBudget, f, d, maxBudget, batchSize, numEvalsPerGrad, minSamples, discountRate=discountRate,
                                          a=a, c=c, useTqdm=useTqdm, useSPSA=useSPSA)
        return results


class restlessTests:

    # params is [f, k, d, maxBudget, batchSize, numEvalsPerGrad]
    def restlessOCBA(sharedParams, discountFactor, windowLength, minSamples,
                                a=.001, c=.001, startPos=False, useTqdm=True, useSPSA=False):
        # UCBSearch(f, k, d, maxBudget, minSamples, batchSize, numEvalsPerGrad)
        f = sharedParams[0]
        k = sharedParams[1]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = restlessBandits.restlessSearch(baiAllocations.discountedOCBA.getBudget, discountFactor, windowLength,
                                                 f, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                          a=a, c=c, startPos=startPos, useTqdm=useTqdm, useSPSA=useSPSA)
        return results

    def restlessInfiniteOCBA(sharedParams, discountFactor, windowLength, minSamples,
                                    a=.001, c=.001, useTqdm=True, useSPSA=False):
        f = sharedParams[0]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = restlessBandits.restlessInfiniteSearch(baiAllocations.discountedOCBA.getBudget, discountFactor, windowLength,
                                                         f, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                                   a=a, c=c, useTqdm=useTqdm, useSPSA=useSPSA)
        return results

    def restlessUCB(sharedParams, discountFactor, windowLength, minSamples,
                                a=.001, c=.001, startPos=False, useTqdm=True, useSPSA=False):
        # UCBSearch(f, k, d, maxBudget, minSamples, batchSize, numEvalsPerGrad)
        f = sharedParams[0]
        k = sharedParams[1]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = restlessBandits.restlessSearch(baiAllocations.discountedUCB.getBudget, discountFactor, windowLength,
                                                 f, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                          a=a, c=c, startPos=startPos, useTqdm=useTqdm, useSPSA=useSPSA)
        return results

    def restlessInfiniteUCB(sharedParams, discountFactor, windowLength, minSamples,
                                    a=.001, c=.001, useTqdm=True, useSPSA=False):
        f = sharedParams[0]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = restlessBandits.restlessInfiniteSearch(baiAllocations.discountedUCB.getBudget, discountFactor, windowLength,
                                                         f, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                                   a=a, c=c, useTqdm=useTqdm, useSPSA=useSPSA)
        return results


class tradTests:

    def tradOCBA(sharedParams, minSamples,
                           a=.001, c=.001, startPos=False, useTqdm=True, useSPSA=False):
        f = sharedParams[0]
        k = sharedParams[1]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = tradBandits.tradSearch(baiAllocations.OCBA.getBudget, f, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                          a=a, c=c, startPos=startPos, useTqdm=useTqdm, useSPSA=useSPSA)
        return results

    def tradInfiniteOCBA(sharedParams, minSamples,
                                   a=.001, c=.001, useTqdm=True, useSPSA=False):
        f = sharedParams[0]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = tradBandits.tradInfiniteSearch(baiAllocations.OCBA.getBudget, f, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                                     a=a, c=c, useTqdm=useTqdm, useSPSA=useSPSA)
        return results

    def tradUCB(sharedParams, minSamples,
                           a=.001, c=.001, startPos=False, useTqdm=True, useSPSA=False):
        f = sharedParams[0]
        k = sharedParams[1]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = tradBandits.tradSearch(baiAllocations.UCB.getBudget, f, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                          a=a, c=c, startPos=startPos, useTqdm=useTqdm, useSPSA=useSPSA)
        return results

    def tradInfiniteUCB(sharedParams, minSamples,
                                   a=.001, c=.001, useTqdm=True, useSPSA=False):
        f = sharedParams[0]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = tradBandits.tradInfiniteSearch(baiAllocations.UCB.getBudget, f, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                                     a=a, c=c, useTqdm=useTqdm, useSPSA=useSPSA)
        return results


class otherTests:

    def uniform(sharedParams,
                           a=.001, c=.001, startPos=False, useTqdm=True, useSPSA=False):
        f = sharedParams[0]
        k = sharedParams[1]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = uniformAlloc.uniformSearch(f, k, d, maxBudget, batchSize, numEvalsPerGrad,
                                             a=a, c=c, startPos=startPos, useTqdm=useTqdm, useSPSA=useSPSA)
        print("Convergence Dictionary: ", end="")
        print(results[2])
        print()

        instances = results[3]
        for i in range(k):
            print(f"Instance #{i}: ", end="")
            print(instances[i])
            print()
        print("Sample Allocation: ", end="")
        print(results[4])
        return results


    def metaMax(sharedParams,
                           a=.001, c=.001, startPos=False, useTqdm=True, useSPSA=False):
        f = sharedParams[0]
        k = sharedParams[1]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = metaMaxAlloc.metaMaxSearch(f, k, d, maxBudget, batchSize, numEvalsPerGrad,
                                             a=a, c=c, startPos=startPos, useTqdm=useTqdm, useSPSA=useSPSA)
        print("Convergence Dictionary: ", end="")
        print(results[2])
        print()

        instances = results[3]
        for i in range(k):
            print(f"Instance #{i}: ", end="")
            print(instances[i])
            print()
        print("Sample Allocation: ", end="")
        print(results[4])
        return results

    def metaMaxInfinite(sharedParams,
                                   a=.001, c=.001, useTqdm=True, useSPSA=False):
        f = sharedParams[0]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        numEvalsPerGrad = sharedParams[5]
        results = metaMaxAlloc.metaMaxInfiniteSearch(f, d, maxBudget, numEvalsPerGrad,
                                                     a=a, c=c, useTqdm=useTqdm, useSPSA=useSPSA)
        print("Convergence Dictionary: ", end="")
        print(results[2])
        print()

        instances = results[3]
        for i in range(len(instances)):
            print(f"Instance #{i}: ", end="")
            print(instances[i])
            print()
        print("Sample Allocation: ", end="")
        print(results[4])
        return results


# return (xHats[maxIndex], fHats[maxIndex], convergeDic, instances, numSamples, sampleDic)
# colors is k-array with colors of function
    def test_display1DResults(results, fun, colors, fColor ='b', lineWidth=3):
        instances = results[3]
        sampleDic = results[5]
        convergeDic = results[2]
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        plt.subplots_adjust(hspace=.5)
        gradientAllocation.displayInstances1D(fun, instances, ax1, colors, fColor=fColor, lineWidth=lineWidth)
        gradientAllocation.displaySamplingHistory(sampleDic, ax2, colors)
        gradientAllocation.displayMinimaHistory(convergeDic, ax3)

    def test_display3DResults(results, fun, colors, fColor = 'b', lineWidth=3, alpha=.1, showFunction=True, fig=plt.figure()):
        instances = results[3]
        sampleDic = results[5]
        convergeDic = results[2]
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(224)
        # fig, (ax1, ax2, ax3) = plt.subplots(3)
        plt.subplots_adjust(hspace=.5)
        gradientAllocation.displayInstances3D(fun, [0,1], instances, ax1, colors, fColor=fColor, lineWidth=lineWidth, alpha=alpha, showFunction=showFunction)
        gradientAllocation.displaySamplingHistory(sampleDic, ax2, colors)
        gradientAllocation.displayMinimaHistory(convergeDic, ax3)


    def displayNDResults(results, colors, fig=plt.figure()):
        instances = results[3]
        sampleDic = results[5]
        convergeDic = results[2]
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(224)
        # fig, (ax1, ax2, ax3) = plt.subplots(3)
        plt.subplots_adjust(hspace=.5)
        gradientAllocation.displayInstancesND(instances, ax1, colors)
        gradientAllocation.displaySamplingHistory(sampleDic, ax2, colors)
        gradientAllocation.displayMinimaHistory(convergeDic, ax3)


# fun = functions.ackley_adjusted
# fun = functions.griewank_adjusted
# fun = functions.ackley_adjusted
fun = functions.rastrigin_adjusted

k = 5
d = 2
maxBudget = 10000
batchSize = 20
numEvalsPerGrad = 2
sharedParams = [fun, k, d, maxBudget, batchSize, numEvalsPerGrad]
minSamples = 2*d+5
a = .001
c = .000001
sharedStartPos = gradientAllocation.stratifiedSampling(d, k)
useSPSA = True

discountFactor = .9
slidingWindow = 15

# """

resultList = []
figList = []

resultsFitOCBA = fitTests.fitOCBA(sharedParams, minSamples, a=a, c=c, startPos=sharedStartPos, useSPSA=True)
figFitOCBA = plt.figure(1)
figFitOCBA.suptitle("Fit OCBA Allocation")
resultList.append(resultsFitOCBA)
figList.append(figFitOCBA)
#
# resultsFitInfiniteOCBA = fitTests.fitInfiniteOCBA(sharedParams, minSamples, a=a, c=c, useSPSA=True)
# figFitInfiniteOCBA = plt.figure(2)
# figFitInfiniteOCBA.suptitle("Fit Infinite OCBA Allocation")
# resultList.append(resultsFitInfiniteOCBA)
# figList.append(figFitInfiniteOCBA)
#
# resultsFitUCB = fitTests.fitUCB(sharedParams, minSamples, a=a, c=c, startPos=sharedStartPos, useSPSA=True)
# figFitUCB = plt.figure(3)
# figFitUCB.suptitle("Fit UCB Allocation")
# resultList.append(resultsFitUCB)
# figList.append(figFitUCB)
#
# resultsFitInfiniteUCB = fitTests.fitInfiniteUCB(sharedParams, minSamples, a=a, c=c, useSPSA=True)
# figFitInfiniteUCB = plt.figure(4)
# figFitInfiniteUCB.suptitle("Fit Infinite UCB Allocation")
# resultList.append(resultsFitInfiniteUCB)
# figList.append(figFitInfiniteUCB)
#
# resultsRestlessOCBA = restlessTests.restlessOCBA(sharedParams, discountFactor, slidingWindow, minSamples, a=a, c=c, startPos=sharedStartPos, useSPSA=True)
# figRestlessOCBA = plt.figure(5)
# figRestlessOCBA.suptitle("Restless OCBA Allocation")
# resultList.append(resultsRestlessOCBA)
# figList.append(figRestlessOCBA)
#
# resultsRestlessInfiniteOCBA = restlessTests.restlessInfiniteOCBA(sharedParams, discountFactor, slidingWindow, minSamples, a=a, c=c, useSPSA=True)
# figRestlessInfiniteOCBA = plt.figure(6)
# figRestlessInfiniteOCBA.suptitle("Restless Infinite OCBA Allocation")
# resultList.append(resultsRestlessInfiniteOCBA)
# figList.append(figRestlessInfiniteOCBA)
#
# resultsRestlessUCB = restlessTests.restlessUCB(sharedParams, discountFactor, slidingWindow, minSamples, a=a, c=c, startPos=sharedStartPos, useSPSA=True)
# figRestlessUCB = plt.figure(7)
# figRestlessUCB.suptitle("Restless UCB Allocation")
# resultList.append(resultsRestlessUCB)
# figList.append(figRestlessUCB)
#
# resultsRestlessInfiniteUCB = restlessTests.restlessInfiniteUCB(sharedParams, discountFactor, slidingWindow, minSamples, a=a, c=c, useSPSA=True)
# figRestlessInfiniteUCB = plt.figure(8)
# figRestlessInfiniteUCB.suptitle("Restless Infinite UCB Allocation")
# resultList.append(resultsRestlessInfiniteUCB)
# figList.append(figRestlessInfiniteUCB)
#
#
# resultsTradOCBA = tradTests.tradOCBA(sharedParams, minSamples, a=a, c=c, startPos=sharedStartPos, useSPSA=True)
# figTradOCBA = plt.figure(9)
# figTradOCBA.suptitle("Trad OCBA Allocation")
# resultList.append(resultsTradOCBA)
# figList.append(figTradOCBA)
#
# resultsTradInfiniteOCBA = tradTests.tradInfiniteOCBA(sharedParams, minSamples, a=a, c=c, useSPSA=True)
# figTradInfiniteOCBA = plt.figure(10)
# figTradInfiniteOCBA.suptitle("Trad Infinite OCBA Allocation")
# resultList.append(resultsTradInfiniteOCBA)
# figList.append(figTradInfiniteOCBA)
#
# resultsTradUCB = tradTests.tradUCB(sharedParams, minSamples, a=a, c=c, startPos=sharedStartPos, useSPSA=True)
# figTradUCB = plt.figure(11)
# figTradUCB.suptitle("Trad UCB Allocation")
# resultList.append(resultsTradUCB)
# figList.append(figTradUCB)
#
# resultsTradInfiniteUCB = tradTests.tradInfiniteUCB(sharedParams, minSamples, a=a, c=c, useSPSA=True)
# figTradInfiniteUCB = plt.figure(12)
# figTradInfiniteUCB.suptitle("Trad Infinite UCB Allocation")
# resultList.append(resultsTradInfiniteUCB)
# figList.append(figTradInfiniteUCB)
#
#
# resultsUniform = otherTests.uniform(sharedParams, a=a, startPos=sharedStartPos, useSPSA=True)
# figUniform = plt.figure(13)
# figUniform.suptitle("Uniform Allocation")
# resultList.append(resultsUniform)
# figList.append(figUniform)
#
# resultsMetaMax = otherTests.metaMax(sharedParams, a=a, startPos=sharedStartPos, useSPSA=True)
# figMetaMax = plt.figure(14)
# figMetaMax.suptitle("MetaMax Allocation")
# resultList.append(resultsMetaMax)
# figList.append(figMetaMax)
#
# resultsMetaMaxInfinite = otherTests.metaMaxInfinite(sharedParams, a=a, useSPSA=True)
# figMetaMaxInfinite = plt.figure(15)
# figMetaMaxInfinite.suptitle("MetaMax Infinite Allocation")
# resultList.append(resultsMetaMaxInfinite)
# figList.append(figMetaMaxInfinite)


# convergeDic = resultsMetaMaxInfinite[2]

# convergeKeys = list(convergeDic.keys())
# for i in range(len(convergeKeys)-1):
#     if convergeDic[convergeKeys[i]] < convergeDic[convergeKeys[i+1]]:
#         print(f"{i}th key")

# """
yRange = [-1, 6]
# colors=['g','r','c','y','m','k','brown','orange','purple','pink']
colors = [(random.random(), random.random(), random.random()) for i in range(1000)]
lineWidth = 2.5
alpha = .1


# testing = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
for i in range(len(resultList)):
    visualize.display3DResults(resultList[i], fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figList[i])

# test.test_display3DResults(resultsFitOCBA, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figFitOCBA)
# test.test_display3DResults(resultsRestlessOCBA, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figRestlessOCBA)
# test.test_display3DResults(resultsTradOCBA, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figTradOCBA)
# test.test_display3DResults(resultsTradOCBA1, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figTradOCBA1)

# test.test_display3DResults(resultsTradOCBAInfinite, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figTradOCBAInfinite)
# test.test_display3DResults(resultsFitUCB, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figFitUCB)
# test.test_display3DResults(resultsTradUCB, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figTradUCB)
# test.test_display3DResults(resultsTradUCB1, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figTradUCB1)

# test.test_display3DResults(resultsTradUCBInfinite, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figTradUCBInfinite)

# test.test_display3DResults(resultsUniform, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figUniform)
# test.test_display3DResults(resultsMetaMax, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figMetaMax)
# test.test_display3DResults(resultsMetaMaxInfinite, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figMetaMaxInfinite)

# test.test_displayNDResults(resultsOCBA,colors,fig=figOCBA)
# test.test_displayNDResults(resultsUCB,colors,fig=figUCB)
# test.test_displayNDResults(resultsUniform,colors,fig=figUniform)
# test.test_displayNDResults(resultsMetaMax,colors,fig=figMetaMax)
#
#
plt.show()

# """

"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
functions.display3D(functions.rastrigin_adjusted, (0, 1), ax)
plt.show()
# """