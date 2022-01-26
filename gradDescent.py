import math
import numpy as np
import matplotlib
import random
from scipy import optimize


class methods:

    def get_at(t, stability = 60, alpha=.602, a=.001):
        denom = (stability+t+1)**alpha
        return a/denom

    def get_ct(t, gamma = .101, c=.001):
        denom = (t+1) ** gamma
        return c/denom

    def partials(f, x, t, c=.001):
        c_t = methods.get_ct(t, c=c)
        d = len(x)
        grad = [0] * d
        for dim in range(d):
            # try:
            #     x1 = x.copy()
            # except AttributeError:
            #     pass
            x1 = x.copy()

            x1[dim] += c_t
            x2 = x.copy()
            x2[dim] -= c_t
            numer = f(x1) - f(x2)
            grad[dim] = numer / (2 * c_t)

        # grad /= ((grad**2).sum*()**.5)
        return np.array(grad)

    def step(x, t, partials, a=.001):
        a_t = methods.get_at(t, a=a)
        return np.add(x, np.multiply(partials, a_t))


    def descend(point, f, t, a, c, iters, q):
        newPoints = []
        for iter in range(iters):
            partials = methods.partials(f, point[0], t, c=c)
            partials = np.negative(partials)
            newX = methods.step(point[0], t, partials, a=a)
            newPoints.append((newX, f(newX)))
            t += 1

        q.put(newPoints)


class finiteDifs:
    def __init__(self, alpha=.602, gamma=.101, stability=60, a=.001, c=.001):
        self.alpha = alpha
        self.gamma = gamma
        self.stability = stability
        self.a = a
        self.c = c

    def get_at(self, t):
        denom = (self.stability+t+1)**self.alpha
        return self.a/denom

    def get_ct(self, t):
        denom = (t+1) ** self.gamma
        return self.c/denom

# for the analytic functions, try doing a direct gradient
    def partials(self, f, x, t):
        c_t = self.get_ct(t)
        d = len(x)
        grad = [0]*d
        for dim in range(d):
            # try:
            #     x1 = x.copy()
            # except AttributeError:
            #     pass
            x1 = x.copy()

            x1[dim] += c_t
            x2 = x.copy()
            x2[dim] -= c_t
            numer = f(x1) - f(x2)
            grad[dim] = numer/(2*c_t)

        # grad /= ((grad**2).sum*()**.5)
        return np.array(grad)


#        return np.array(optimize.approx_fprime(x, f, c_t))

    def step(self, x, t, partials):
        a_t = self.get_at(t)
        return np.add(x, np.multiply(partials, a_t))

    # x is starting point
    def gradDescent(self, f, x, steps):
        min = f(x)
        minParams = x
        samples = [(x, min)]
        for t in range(steps):
            grad = self.partials(f, x, t+1)
            x = self.step(x, t+1, np.negative(grad))
            eval = f(x)
            samples.append((x, eval))
            if eval < min:
                min = f(x)
                minParams = x
        return minParams, min, samples

    def gradAscent(self, f, x, steps):
        max = f(x)
        maxParams = x
        samples = [(x, max)]
        for t in range(steps):
            grad = self.partials(f, x, t + 1)
            x = self.step(x, t + 1, grad)
            eval = f(x)
            samples.append(x, eval)
            if eval > max:
                max = f(x)
                maxParams = x
        return maxParams, max, samples


class SPSA(finiteDifs):

    # b is B_t, random vector
    # d is number of dimensions
    def get_b(self, d):
        return np.array([random.randint(0, 1)*2-1 for _ in range(d)])


    # f is the function to minimize
    # x is input vector (numpy array), d dimensions
    # t-th step
    # c is for get_ct
    # returns an approximation of the gradient
    def partials(self, f, x, t):
        d = len(x)
        c_t = self.get_ct(t)
        b = self.get_b(d)
        pert = np.multiply(b, c_t)
        numer = f(np.add(x, pert)) - f(np.subtract(x, pert))

        partials = [None]*d
        for l in range(d):
            denom = 2*c_t*b[l]
            partials[l] = numer/denom

        return np.array(partials)


# class STAR_SPSA(finiteDifs):

