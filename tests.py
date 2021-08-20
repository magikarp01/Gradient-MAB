import unittest
import fitAlloc
import kriging
import gradDescent
import functions
import matplotlib.pyplot as plt

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

    def test_OCBAFit(self):
        # OCBA_Budget(f, k, d, maxBudget, minSamples, batchSize, numEvalsPerGrad)
        k = 5
        d = 1
        maxBudget = 10000
        minSamples = 10
        batchSize = 20
        numEvalsPerGrad = 2*d
        results = fitAlloc.OCBA_Budget(functions.min3Parabola, k, d, maxBudget, minSamples, batchSize, numEvalsPerGrad, discountRate=.95, a=.05)
        # return (xHats[maxIndex], fHats[maxIndex], convergeDic, instances, numSamples, sampleDic)
        print("Convergence Dictionary: ", end="")
        print(results[2])
        print()

        instances = results[3]
        sampleDic = results[5]
        for i in range(k):
            print(f"Instance #{i}: ", end="")
            print(instances[i])
            print()
        print("Sample Allocation: ", end="")
        print(results[4])
        fig, (ax1, ax2) = plt.subplots(2)
        colors = ['b','g','r','c','y']
        fitAlloc.displayInstances1D(functions.min3Parabola, instances, ax1, colors)
        fitAlloc.displaySamplingHistory(sampleDic, ax2, colors)
        plt.show()

# if __name__ == '__main__':
#     unittest.main()

test = TestClass()
test.test_OCBAFit()


