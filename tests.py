import unittest
import random

import UCBAlloc
import kriging
import gradDescent
import functions
import matplotlib.pyplot as plt
import gradientAllocation

import OCBAAlloc
import uniformAlloc
import metaMaxAlloc

import fitBandits
import restlessBandits
import tradBandits
import baiAllocations

# class TestClass(unittest.TestCase):
class fitTests:

    # params is [f, k, d, maxBudget, batchSize, numEvalsPerGrad]
    def fitOCBA(sharedParams, minSamples,
                           discountRate=.9, a=.001, c=.001, startPos=False, useTqdm=True, useSPSA=False):
        # UCBSearch(f, k, d, maxBudget, minSamples, batchSize, numEvalsPerGrad)
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
        results = fitBandits.fitSearch(baiAllocations.OCBA.getBudget, f, d, maxBudget, batchSize, numEvalsPerGrad, minSamples, discountRate=discountRate,
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

    def fitInfiniteOCBA(sharedParams, minSamples,
                           discountRate=.9, a=.001, c=.001, startPos=False, useTqdm=True, useSPSA=False):
        # UCBSearch(f, k, d, maxBudget, minSamples, batchSize, numEvalsPerGrad)
        f = sharedParams[0]
        k = sharedParams[1]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = fitBandits.fitSearch(baiAllocations.UCB.getBudget, f, d, maxBudget, batchSize, numEvalsPerGrad, minSamples, discountRate=discountRate,
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
        # return (xHats[maxIndex], fHats[maxIndex], convergeDic, instances, numSamples, sampleDic)
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

    def restlessUCB(sharedParams, discountFactor, windowLength, minSamples,
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
        # return (xHats[maxIndex], fHats[maxIndex], convergeDic, instances, numSamples, sampleDic)
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

    def restlessInfiniteUCB(sharedParams, discountFactor, windowLength, minSamples,
                                    a=.001, c=.001, useTqdm=True, useSPSA=False):
        f = sharedParams[0]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = restlessBandits.restlessInfiniteSearch(baiAllocations.discountedOCBA.getBudget, discountFactor, windowLength,
                                                         f, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
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
        # return (xHats[maxIndex], fHats[maxIndex], convergeDic, instances, numSamples, sampleDic)
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

    def tradInfiniteOCBA(self, sharedParams, minSamples,
                                   a=.001, c=.001, useTqdm=True, useSPSA=False):
        f = sharedParams[0]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = OCBAAlloc.tradOCBAInfiniteSearch(f, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
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

    def tradUCB(sharedParams, minSamples,
                           a=.001, c=.001, startPos=False, useTqdm=True, useSPSA=False):
        f = sharedParams[0]
        k = sharedParams[1]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = tradBandits.tradSearch(baiAllocations.OCBA.getBudget, f, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                          a=a, c=c, startPos=startPos, useTqdm=useTqdm, useSPSA=useSPSA)
        # return (xHats[maxIndex], fHats[maxIndex], convergeDic, instances, numSamples, sampleDic)
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

    def tradInfiniteUCB(self, sharedParams, minSamples,
                                   a=.001, c=.001, useTqdm=True, useSPSA=False):
        f = sharedParams[0]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = OCBAAlloc.tradOCBAInfiniteSearch(f, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
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


class otherTests:

    def uniform(self, sharedParams,
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


    def metaMax(self, sharedParams,
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

    def metaMaxInfinite(self, sharedParams,
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
    def test_display1DResults(self, results, fun, colors, fColor ='b', lineWidth=3):
        instances = results[3]
        sampleDic = results[5]
        convergeDic = results[2]
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        plt.subplots_adjust(hspace=.5)
        gradientAllocation.displayInstances1D(fun, instances, ax1, colors, fColor=fColor, lineWidth=lineWidth)
        gradientAllocation.displaySamplingHistory(sampleDic, ax2, colors)
        gradientAllocation.displayMinimaHistory(convergeDic, ax3)

    def test_display3DResults(self, results, fun, colors, fColor = 'b', lineWidth=3, alpha=.1, showFunction=True, fig=plt.figure()):
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


    def test_displayNDResults(self, results, colors, fig=plt.figure()):
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
fun = functions.griewank_adjusted

k = 5
d = 2
maxBudget = 10000
batchSize = 20
numEvalsPerGrad = 2
sharedParams = [fun, k, d, maxBudget, batchSize, numEvalsPerGrad]
test = TestClass()
minSamples = 2*d+5
a = .01
c = .000001
sharedStartPos = gradientAllocation.stratifiedSampling(d, k)
useSPSA = True

discountFactor = .9
slidingWindow = 15

# """
# resultsFitOCBA = test.test_fitOCBASearch(sharedParams, minSamples, a=a, c=c, startPos=sharedStartPos, useSPSA=True)
# figFitOCBA = plt.figure(100)
# figFitOCBA.suptitle("Fit OCBA Allocation")
#
# resultsRestlessOCBA = test.test_restlessOCBASearch(sharedParams, discountFactor, slidingWindow, minSamples, a=a, c=c, startPos=sharedStartPos, useSPSA=True)
# figRestlessOCBA = plt.figure(1000)
# figRestlessOCBA.suptitle("Restless OCBA Allocation")

resultsTradOCBA = test.test_tradOCBASearch(sharedParams, minSamples, a=a, c=c, startPos=sharedStartPos, useSPSA=True)
figTradOCBA = plt.figure(200)
figTradOCBA.suptitle("Traditional OCBA Allocation")

resultsTradOCBA1 = test.test_tradOCBASearch1(sharedParams, minSamples, a=a, c=c, startPos=sharedStartPos, useSPSA=True)
figTradOCBA1 = plt.figure(100)
figTradOCBA1.suptitle("Old traditional OCBA Allocation")


# resultsTradOCBAInfinite = test.test_tradOCBAInfiniteSearch(sharedParams, minSamples, a=a, c=c, useSPSA=True)
# figTradOCBAInfinite = plt.figure(800)
# figTradOCBAInfinite.suptitle("Traditional OCBA Infinite Allocation")

# resultsFitUCB = test.test_fitUCBSearch(sharedParams, minSamples, a=a, c=c, startPos=sharedStartPos, useSPSA=True)
# figFitUCB = plt.figure(300)
# figFitUCB.suptitle("Fit UCB Allocation")

resultsTradUCB = test.test_tradUCBSearch(sharedParams, minSamples, a=a, c=c, startPos=sharedStartPos, useSPSA=True)
figTradUCB = plt.figure(400)
figTradUCB.suptitle("Traditional UCB Allocation")

resultsTradUCB1 = test.test_tradUCBSearch1(sharedParams, minSamples, a=a, c=c, startPos=sharedStartPos, useSPSA=True)
figTradUCB1 = plt.figure(300)
figTradUCB1.suptitle("Old Traditional UCB Allocation")

# resultsTradUCBInfinite = test.test_tradUCBInfiniteSearch(sharedParams, minSamples, a=a, c=c, useSPSA=True)
# figTradUCBInfinite = plt.figure(900)
# figTradUCBInfinite.suptitle("Traditional UCB Infinite Allocation")

# resultsUniform = test.test_uniformSearch(sharedParams, a=a, startPos=sharedStartPos, useSPSA=True)
# figUniform = plt.figure(500)
# figUniform.suptitle("Uniform Allocation")
#
# resultsMetaMax = test.test_metaMaxSearch(sharedParams, a=a, startPos=sharedStartPos, useSPSA=True)
# figMetaMax = plt.figure(600)
# figMetaMax.suptitle("MetaMax Allocation")


# resultsMetaMaxInfinite = test.test_metaMaxInfiniteSearch(sharedParams, a=a, useSPSA=True)
# figMetaMaxInfinite = plt.figure(700)
# figMetaMaxInfinite.suptitle("MetaMax Infinite Allocation")

# convergeDic = resultsMetaMaxInfinite[2]

# convergeKeys = list(convergeDic.keys())
# for i in range(len(convergeKeys)-1):
#     if convergeDic[convergeKeys[i]] < convergeDic[convergeKeys[i+1]]:
#         print(f"{i}th key")

yRange = [-1, 6]
colors=['g','r','c','y','m','k','brown','orange','purple','pink']
# colors = [(random.random(), random.random(), random.random()) for i in range(1000)]
lineWidth = 2.5
alpha = .1


# test.test_display3DResults(resultsFitOCBA, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figFitOCBA)
# test.test_display3DResults(resultsRestlessOCBA, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figRestlessOCBA)
test.test_display3DResults(resultsTradOCBA, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figTradOCBA)
test.test_display3DResults(resultsTradOCBA1, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figTradOCBA1)

# test.test_display3DResults(resultsTradOCBAInfinite, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figTradOCBAInfinite)
# test.test_display3DResults(resultsFitUCB, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figFitUCB)
test.test_display3DResults(resultsTradUCB, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figTradUCB)
test.test_display3DResults(resultsTradUCB1, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figTradUCB1)

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

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# functions.display3D(functions.griewank_adjusted, (0, 1), ax)
# plt.show()
