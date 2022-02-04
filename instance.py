import numpy as np
import gradientAllocation


class Instance:


    # makes new instance of search
    # starts at random or ___, performs minSamples descents,
    def __init__(self, f, d, gradDescentObject, startPos = -1):
        self.f = f
        self.d = d
        self.gradDescentObject = gradDescentObject
        if startPos == -1:
            startPos = gradientAllocation.randomParams(d)

        startVal = self.f(startPos)
        # point is (x, f(x))
        self.points = [(startPos, startVal)]

        # list of f(x) for previously visited x
        self.pointValues = [startVal]

        # for gradientDescent
        self.numSamples = 0

    def get_points(self):
        return self.points

    def get_pointValues(self):
        return [point[1] for point in self.points]

    def get_fHat(self):
        return min(self.pointValues)

    def get_xHat(self):
        return np.argmax(self.pointValues)

    # returns (x, f(x))
    def get_lastPoint(self):
        return self.points[-1]

    def get_numSamples(self):
        return self.numSamples

    def get_gradDescentObject(self):
        return self.gradDescentObject

    # should speed up multiprocessing
    def multiprocessDescend(self, newPoints):
        for newPoint in newPoints:
            self.points.append(newPoint)
            self.numSamples += 1

    # performs one descent step
    def descend(self):
        partials = self.gradDescentObject.partials(self.f, self.points[-1][0], self.numSamples)
        partials = np.negative(partials)
        newX = self.gradDescentObject.step(self.points[-1][0], self.numSamples, partials)
        self.points.append((newX, self.f(newX)))
        self.pointValues.append(self.get_lastPoint()[1])

        # previously, it was self.numSamples += minEvalsPerGrad
        self.numSamples += 1