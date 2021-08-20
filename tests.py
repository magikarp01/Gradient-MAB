import unittest
import kriging
import gradDescent
import functions
import matplotlib.pyplot as plt
import gradientAllocation

import fitAlloc
import uniformAlloc


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


    # params is [f, k, d, maxBudget, minSamples, batchSize, numEvalsPerGrad]
    def test_OCBASearch(self, sharedParams, minSamples, discountRate=.95, a=.02, c=.001):
        # OCBASearch(f, k, d, maxBudget, minSamples, batchSize, numEvalsPerGrad)
        f = sharedParams[0]
        k = sharedParams[1]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = fitAlloc.OCBASearch(f, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples, discountRate=discountRate, a=a, c=c)
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


    def test_uniformSearch(self, sharedParams, a=.02, c=.001):
        f = sharedParams[0]
        k = sharedParams[1]
        d = sharedParams[2]
        maxBudget = sharedParams[3]
        batchSize = sharedParams[4]
        numEvalsPerGrad = sharedParams[5]
        results = uniformAlloc.uniformSearch(f, k, d, maxBudget, batchSize, numEvalsPerGrad, a=a, c=c)
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

    def test_displayResults(self, results, colors=['b','g','r','c','y']):
        instances = results[3]
        sampleDic = results[5]
        fig, (ax1, ax2) = plt.subplots(2)
        gradientAllocation.displayInstances1D(functions.min3Parabola, instances, ax1, colors)
        gradientAllocation.displaySamplingHistory(sampleDic, ax2, colors)
        plt.show()



# if __name__ == '__main__':
#     unittest.main()

f = functions.min3Parabola
k = 5
d = 1
maxBudget = 2500
batchSize = 20
numEvalsPerGrad = 2*d
sharedParams = [f, k, d, maxBudget, batchSize, numEvalsPerGrad]

test = TestClass()
minSamples = 10
results = test.test_OCBASearch(sharedParams, minSamples, a=.001)
# results = test.test_uniformSearch(sharedParams, a=.001)
test.test_displayResults(results)
