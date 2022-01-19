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



    def get_points(self):
        return self.points

    def get_fHat(self):
        return max(self.pointValues)

    def get_xHat(self):
        return np.argmax(self.pointValues)

    # returns (x, f(x))
    def get_lastPoint(self):
        return self.points[-1]

    # performs one descent step
    def descend(self, f, a, c):

        partials = self.gradDescentObject.partials(f, self.points[-1][0], self.numSamples, c=c)
        partials = np.negative(partials)
        newX = self.gradDescentObject.step(self.points[-1][0], self.numSamples, partials, a=a)
        self.points.append((newX, f(newX)))

        self.numSamples += 1
