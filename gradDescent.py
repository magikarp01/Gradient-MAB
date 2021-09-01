import math
import numpy as np
import matplotlib
import random
from scipy import optimize


class finiteDifs:
    def __init__(self, alpha=.602, gamma=.101, stability=60):
        self.alpha = alpha
        self.gamma = gamma
        self.stability = stability

    def get_at(self, a, t):
        denom = (self.stability+t+1)**self.alpha
        return a/denom

    def get_ct(self, c, t):
        denom = (t+1) ** self.gamma
        return c/denom

# for the analytic functions, try doing a direct gradient
    def partials(self, f, x, t, c=.001):
        c_t = self.get_ct(c, t)
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

        return np.array(grad)


        return np.array(optimize.approx_fprime(x, f, c_t))

    def step(self, x, t, partials, a=.001):
        a_t = self.get_at(a, t)
        return np.add(x, np.multiply(partials, a_t))

    # x is starting point
    def gradDescent(self, f, x, steps, a=.001, c=.001):
        min = f(x)
        minParams = x
        samples = [(x, min)]
        for t in range(steps):
            grad = self.partials(f, x, t+1, c)
            x = self.step(x, t+1, np.negative(grad), a)
            eval = f(x)
            samples.append((x, eval))
            if eval < min:
                min = f(x)
                minParams = x
        return minParams, min, samples

    def gradAscent(self, f, x, steps, a=.001, c=.001):
        max = f(x)
        maxParams = x
        samples = [(x, max)]
        for t in range(steps):
            grad = self.partials(f, x, t + 1, c)
            x = self.step(x, t + 1, grad, a)
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
    def partials(self, f, x, t, c=.001):
        d = len(x)
        c_t = self.get_ct(c, t)
        b = self.get_b(d)
        pert = np.multiply(b, c_t)
        numer = f(np.add(x, pert)) - f(np.subtract(x, pert))

        partials = [None]*d
        for l in range(d):
            denom = 2*c_t*b[l]
            partials[l] = numer/denom

        return np.array(partials)


# class STAR_SPSA(finiteDifs):

