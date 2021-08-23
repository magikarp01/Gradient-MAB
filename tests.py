import unittest
import kriging
import gradDescent
import functions
import matplotlib.pyplot as plt
import gradientAllocation

import fitAlloc
import uniformAlloc
import metaMaxAlloc


# class TestClass(unittest.TestCase):
class TestClass:

    def test_gradDescent(self):
        fObject = gradDescent.finiteDifs()
        f = functions.min3Parabola
        startPos = fitAlloc.randomParams(1)
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
        print(fitAlloc.stratifiedSampling(3, 27))

    def test_kroneckers(self):
        values = [1, 2, 3, 5, 209]
        kroneckers = fitAlloc.getKroneckers(values)
        print(kroneckers)

    def test_budget(self):
        values = [1, 2, 3, 5, 209]
        kroneckers = fitAlloc.getKroneckers(values)
        variances = [3, 4, 1, 2, 4]
        numSamples = [10, 6, 5, 3, 8]
        budget = fitAlloc.getBudget(values, variances, kroneckers, numSamples)
        print(budget)


    # params is [f, k, d, maxBudget, batchSize, numEvalsPerGrad]
    def test_OCBASearch(self, sharedParams, minSamples,
                        discountRate=.8, a=.001, c=.001, startPos=False):
        # OCBASearch(f, k, d, maxBudget, minSamples, batchSize, numEvalsPerGrad)
        f = sharedParams[0]
        k = sharedParams[1]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = fitAlloc.OCBASearch(f, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples, discountRate=discountRate,
                                      a=a, c=c, startPos=False)
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
                           a=.001, c=.001, startPos=False):
        f = sharedParams[0]
        k = sharedParams[1]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = uniformAlloc.uniformSearch(f, k, d, maxBudget, batchSize, numEvalsPerGrad,
                                             a=a, c=c, startPos=startPos)
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
                           a=.001, c=.001, startPos=False):
        f = sharedParams[0]
        k = sharedParams[1]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = metaMaxAlloc.metaMaxSearch(f, k, d, maxBudget, batchSize, numEvalsPerGrad,
                                             a=a, c=c, startPos=startPos)
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


"""
fun = functions.ackley_adjusted
k = 5
d = 2
maxBudget = 10000
batchSize = 100
numEvalsPerGrad = 2*d
sharedParams = [fun, k, d, maxBudget, batchSize, numEvalsPerGrad]
test = TestClass()
minSamples = 10
a = .002
c = .000001
sharedStartPos = gradientAllocation.stratifiedSampling(d, k)

resultsOCBA = test.test_OCBASearch(sharedParams, minSamples, a=a, c=c, startPos=sharedStartPos)
resultsUniform = test.test_uniformSearch(sharedParams, a=a, startPos=sharedStartPos)
resultsMetaMax = test.test_metaMaxSearch(sharedParams, a=a, startPos=sharedStartPos)

yRange = [-1, 6]
colors=['g','r','c','y','m','k','brown','orange','purple','pink']
lineWidth = 2.5
alpha = .1

figOCBA = plt.figure(100)
figOCBA.suptitle("OCBA Allocation")
figUniform = plt.figure(200)
figUniform.suptitle("Uniform Allocation")
figMetaMax = plt.figure(300)
figMetaMax.suptitle("MetaMax Allocation")

test.test_display3DResults(resultsOCBA, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figOCBA)
test.test_display3DResults(resultsUniform, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figUniform)
test.test_display3DResults(resultsMetaMax, fun, colors, fColor = 'b', lineWidth=lineWidth, alpha=alpha, showFunction=True, fig=figMetaMax)


plt.show()

"""

# functions.display3D(functions.ackley_adjusted, (0, 1))

