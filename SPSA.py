import math

import numpy as np
import matplotlib
import random

alpha = 0.602
gamma = 0.101
stability = 60

# a has to be derived empirically
def get_at(a, t):
    denom = (stability+t+1)**alpha
    return a/denom


# c has to be derived empirically
def get_ct(c, t):
    denom = (t+1)**gamma
    return c/denom


# b is B_t, random vector
# d is number of dimensions
def get_b(d):
    return np.array([random.randint(0, 1)*2-1 for _ in range(d)])


# f is the function to minimize
# x is input vector (numpy array), d dimensions
# t-th step
# c is for get_ct
# returns an approximation of the gradient
def partials(f, x, d, t, c):
    c_t = get_ct(c, t)
    b = get_b(d)
    pert = np.multiply(b, c_t)
    numer = f(np.add(x, pert)) - f(np.subtract(x, pert))

    partials = [None]*d
    for l in range(d):
        denom = 2*c_t*b[l]
        partials[l] = numer/denom

    return np.array(partials)


def step(x, t, partials, a):
    a_t = get_at(a, t)
    return np.add(x, np.multiply(partials, a_t))


def f(x):
    sum = 0
    for a in x:
        sum += abs(a)
    return 100-sum


# x is starting point
def gradDescent(f, x, d, steps, a, c):
    max = f(x)
    maxParams = x
    for t in range(steps):
        grad = partials(f, x, d, t+1, c)
        x = step(x, t+1, grad, a)
        if f(x) > max:
            max = f(x)
            maxParams = x
    return (maxParams, max)


#print(gradDescent(f, [1000]*5, 5, 10000, 100, .2))

# x = np.array([1000]*5)
# max = f(x)
# maxParams = x
# for t in range(10000):
# #    print(get_b(5))
#     grad = partials(f, x, 5, t+1, .2)
#     x = step(x, t+1, grad, 100)
#     # print(grad)
#     # print(x)
#     # print(f(x))
#     # print()
#     if f(x) > max:
#         max = f(x)
#         maxParams = x
#
# print(max)
# print(maxParams)