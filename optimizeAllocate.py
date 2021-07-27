# a general class for allocating a budget for optimization
# other classes will extend this class

import numpy as np
import matplotlib
import math
import random
from scipy.spatial import ConvexHull
import gradDescent


class optimizeAllocation:

    SPSAObject = gradDescent.SPSA()
    finiteDifsObject = gradDescent.finiteDifs()
    from tqdm import tqdm

    # all params are in range [0, 1)
    def randomParams(d):
        vec = [None]*d
        for i in range(d):
            vec[i] = random.random()
        return np.array(vec)

