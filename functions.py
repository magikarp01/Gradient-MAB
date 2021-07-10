import math
import numpy as np
import SPSA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# x is input vector, d dimensions
def ackley(x, a = 20, b=.2, c=2*math.pi):
    d = len(x)
    sumSquared = 0
    sumCos = 0
    for i in x:
        sumSquared += i**2
        sumCos += math.cos(c*i)

    t1 = a * math.exp(-b * math.sqrt(1/d * sumSquared))
    t2 = math.exp(1/d * sumCos)
    value = -t1 - t2 + a + math.e
    return value


# ackley_adjusted accepts input with x_i in [0, 1], transforms into [-32.768, 32.768]
def ackley_adjusted(x_adjusted, a = 20, b=.2, c=2*math.pi):
    x = []
    for i in x_adjusted:
        x.append((i-.5)*2*32.768)
    return ackley(np.array(x), a, b, c)


# griewank is best for ranges of [-10, 10]
def griewank(x):
    d = len(x)
    sumSquared = 0
    prodCos = 1
    for j in range(d):
        sumSquared += x[j]**2
        prodCos *= math.cos(x[j] / math.sqrt(j+1))

    sumSquared /= 4000
    return sumSquared - prodCos + 1


# accepts input in [0, 1]
def griewank_adjusted(x_adjusted):
    x = []
    for i in x_adjusted:
        x.append((i-.5)*2*10)
    return griewank(np.array(x))



def display3D(fun, range):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(range[0], range[1], 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([fun(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()