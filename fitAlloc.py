import numpy as np
import matplotlib
import math
import random
from scipy.spatial import ConvexHull
import gradDescent
import kriging

def randomParams(d):
    vec = [None]*d
    for i in range(d):
        vec[i] = random.random()
    return np.array(vec)

def SPSABudget(f, k, d, maxBudget):
    instances = [None]*k
    xHats = [None] * k
    fHats = [None] * k

    for i in range(k):
        startPoint = randomParams(d)
        instances[i] = [startPoint]
        xHats[i] = startPoint
        fHats[i] = f(startPoint)


