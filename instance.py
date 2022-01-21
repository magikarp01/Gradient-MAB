import numpy as np
import gradientAllocation


class Instance:
    # point is (x, f(x))
    points = []

    # list of f(x) for previously visited x
    pointValues = []
    numSamples = 0

    # makes new instance of search
    # starts at random or ___, performs minSamples descents,
    def __init__(self, f, d, gradDescentObject, startPos = -1):
        self.f = f
        self.d = d
        self.gradDescentObject = gradDescentObject
        if startPos == -1:
            startPos = gradientAllocation.randomParams(d)

        startVal = self.f(startPos)
        self.points.append((startPos, startVal))
        self.pointValues.append(startVal)

        # for gradientDescent
        self.t = 0

    def get_points(self):
        return self.points

    def get_pointValues(self):
        return [point[1] for point in self.points]

    def get_fHat(self):
        return max(self.pointValues)

    def get_xHat(self):
        return np.argmax(self.pointValues)

    # returns (x, f(x))
    def get_lastPoint(self):
        return self.points[-1]

    def get_numSamples(self):
        return self.numSamples

    # performs one descent step
    def descend(self, numEvalsPerGrad):

        # previously, it was self.numSamples instead of self.t
        partials = self.gradDescentObject.partials(self.f, self.points[-1][0], self.t)
        partials = np.negative(partials)
        newX = self.gradDescentObject.step(self.points[-1][0], self.t, partials)
        self.points.append((newX, self.f(newX)))

        self.numSamples += numEvalsPerGrad + 2
        self.t += 1