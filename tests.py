import unittest

import UCBAlloc
import kriging
import gradDescent
import functions
import matplotlib.pyplot as plt
import gradientAllocation

import OCBAAlloc
import uniformAlloc
import metaMaxAlloc


# class TestClass(unittest.TestCase):
class TestClass:

    def test_gradDescent(self):
        fObject = gradDescent.finiteDifs()
        f = functions.min3Parabola
        startPos = OCBAAlloc.randomParams(1)
        minima, min, samples = fObject.gradDescent(f, startPos, 1000, a=.01)
        print(f"Starting position is {startPos}")
        print(f"Location of minima is {minima}")
        print(f"Minimum is {min}")
        print(f"Samples are: ", end="")
        print(samples)

    def test_quadFit(self):
        points = [(1,), (2,), (3,), (4,)]
        pointValues = [-1, -4, -9, -16]
        result, variance = kriging.quadEstMin(points, pointValues)
        print(result)
        print(variance)

    def test_stratifiedSampling(self):
        print(OCBAAlloc.stratifiedSampling(3, 27))

    def test_kroneckers(self):
        values = [1, 2, 3, 5, 209]
        kroneckers = OCBAAlloc.getKroneckers(values)
        print(kroneckers)

    def test_budget(self):
        values = [1, 2, 3, 5, 209]
        kroneckers = OCBAAlloc.getKroneckers(values)
        variances = [3, 4, 1, 2, 4]
        numSamples = [10, 6, 5, 3, 8]
        budget = OCBAAlloc.getBudget(values, variances, kroneckers, numSamples)
        print(budget)


    # params is [f, k, d, maxBudget, batchSize, numEvalsPerGrad]
    def test_OCBASearch(self, sharedParams, minSamples,
                        discountRate=.9, a=.001, c=.001, startPos=False, useTqdm=True, useSPSA=False):
        # UCBSearch(f, k, d, maxBudget, minSamples, batchSize, numEvalsPerGrad)
        f = sharedParams[0]
        k = sharedParams[1]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = OCBAAlloc.OCBASearch(f, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples, discountRate=discountRate,
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

    def test_UCBSearch(self, sharedParams, minSamples,
                        discountRate=.9, a=.001, c=.001, startPos=False, useTqdm=True, useSPSA=False):
        # UCBSearch(f, k, d, maxBudget, minSamples, batchSize, numEvalsPerGrad)
        f = sharedParams[0]
        k = sharedParams[1]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = UCBAlloc.UCBSearch(f, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                       discountRate=discountRate,
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

    def test_uniformSearch(self, sharedParams,
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


    def test_metaMaxSearch(self, sharedParams,
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


"""
# fun = functions.ackley_adjusted
fun = functions.griewank_adjusted

k = 5
d = 2
maxBudget = 5000
batchSize = 10
numEvalsPerGrad = 2*d
sharedParams = [fun, k, d, maxBudget, batchSize, numEvalsPerGrad]
test = TestClass()
minSamples = 2*d+5
a = .01
c = .000001
sharedStartPos = gradientAllocation.stratifiedSampling(d, k)
useSPSA = False

resultsOCBA = test.test_OCBASearch(sharedParams, minSamples, a=a, c=c, startPos=sharedStartPos, useSPSA=True)
resultsUCB = test.test_UCBSearch(sharedParams, minSamples, a=a, c=c, startPos=sharedStartPos, useSPSA=True)
resultsUniform = test.test_uniformSearch(sharedParams, a=a, startPos=sharedStartPos, useSPSA=True)
resultsMetaMax = test.test_metaMaxSearch(sharedParams, a=a, startPos=sharedStartPos, useSPSA=True)

yRange = [-1, 6]
colors=['g','r','c','y','m','k','brown','orange','purple','pink']
lineWidth = 2.5
alpha = .1

figOCBA = plt.figure(100)
figOCBA.suptitle("OCBA Allocation")
figUCB = plt.figure(200)
figUCB.suptitle("UCB Allocation")
figUniform = plt.figure(300)
figUniform.suptitle("Uniform Allocation")
figMetaMax = plt.figure(400)
figMetaMax.suptitle("MetaMax Allocation")

test.test_display3DResults(resultsOCBA, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figOCBA)
test.test_display3DResults(resultsUCB, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figUCB)
test.test_display3DResults(resultsUniform, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figUniform)
test.test_display3DResults(resultsMetaMax, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figMetaMax)

# test.test_displayNDResults(resultsOCBA,colors,fig=figOCBA)
# test.test_displayNDResults(resultsUCB,colors,fig=figUCB)
# test.test_displayNDResults(resultsUniform,colors,fig=figUniform)
# test.test_displayNDResults(resultsMetaMax,colors,fig=figMetaMax)
#
#
plt.show()

"""

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
functions.display3D(functions.griewank_adjusted, (0, 1), ax)
plt.show()