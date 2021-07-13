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

    def partials(self, f, x, t, c=.2):
        c_t = self.get_ct(c, t)
        return np.array(optimize.approx_fprime(x, f, c_t))

    def step(self, x, t, partials, a=.16):
        a_t = self.get_at(a, t)
        return np.add(x, np.multiply(partials, a_t))


class SPSA(finiteDifs):

    def __init__(self, alpha = .602, gamma = .101, stability = 60):
        self.alpha = alpha
        self.gamma = gamma
        self.stability = stability

    # a has to be derived empirically
    def get_at(self, a, t):
        denom = (self.stability+t+1)**self.alpha
        return a/denom


    # c has to be derived empirically
    def get_ct(self, c, t):
        denom = (t+1) ** self.gamma
        return c/denom


    # b is B_t, random vector
    # d is number of dimensions
    def get_b(self, d):
        return np.array([random.randint(0, 1)*2-1 for _ in range(d)])


    # f is the function to minimize
    # x is input vector (numpy array), d dimensions
    # t-th step
    # c is for get_ct
    # returns an approximation of the gradient
    def partials(self, f, x, t, c=.2):
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

    def step(self, x, t, partials, a=.16):
        a_t = self.get_at(a, t)
        return np.add(x, np.multiply(partials, a_t))


    # x is starting point
    def gradDescent(self, f, x, steps, a, c):
        min = f(x)
        minParams = x
        for t in range(steps):
            grad = self.partials(f, x, t+1, c)
            x = self.step(x, t+1, grad, a)
            if f(x) < min:
                min = f(x)
                minParams = x
        return (minParams, min)


