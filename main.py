import SPSA
import functions
import metaMax
import math
from scipy.spatial import ConvexHull
import numpy as np

# functions.display3D(functions.griewank, [-5, 5])

# print(metaMax.metaMaxSPSA(functions.reverse_ackley_adjusted, 100, 4, 10000))
print(metaMax.metaMaxSPSABudget(functions.reverse_ackley_adjusted, 100, 4, 1000000))


# x = metaMax.randomParams(3)
# f = functions.ackley_adjusted
# initVal = f(x)
# (minParams, min) = SPSA.gradDescent(f, x, 3, 10000, 1000, 200)

