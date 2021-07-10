import SPSA
import functions
#import metaMax
import math
from scipy.spatial import ConvexHull
import numpy as np

# functions.display3D(functions.griewank, [-5, 5])

for point in ConvexHull([[0, 0], [1,1], [2, 3]]).simplices:
    print(point)

print(np.argmax([0, 2, 5, 1, 3]))

# x = metaMax.randomParams(3)
# f = functions.ackley_adjusted
# initVal = f(x)
# (minParams, min) = SPSA.gradDescent(f, x, 3, 10000, 1000, 200)

