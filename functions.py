import math
import numpy as np
import gradDescent
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time

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
# error is stdev of a random error term, drawn gaussian
def ackley_adjusted(x_adjusted, a = 20, b=.2, c=2 * math.pi, error=0):
    x = []
    for i in x_adjusted:
        x.append((i-.5)*2*32.768)
        # x.append((i-.5)*2*5)

    return ackley(np.array(x), a, b, c)+np.random.normal(0,error)

# supposed to simulate an expensive function
def sleep_ackley_adjusted(x_adjusted, a = 20, b=.2, c=2*math.pi, sleepTime=.5):
    time.sleep(sleepTime)
    return ackley_adjusted(x_adjusted, a, b, c)


# negative
def reverse_ackley_adjusted(x_adjusted, a = 20, b=.2, c=2*math.pi):
    x = []
    for i in x_adjusted:
        x.append((i-.5)*2*32.768)
    return -ackley(np.array(x), a, b, c)


# griewank from multi-start strategies, but negative
# accepts in [-1, 1]
def griewank(x):
    d = len(x)
    sumSquared = 0
    prodCos = 1
    for j in range(d):

        sumSquared += (4 * (math.pi**2) * x[j]**2)/100

        prodCos *= math.cos( 2 * math.pi * x[j] / math.sqrt(j+1))

    return sumSquared - prodCos


# accepts input in [0, 1]
def griewank_adjusted(x_adjusted, error=0):
    x = []
    for i in x_adjusted:
        x.append((i-.5)*2)
    return griewank(np.array(x))+np.random.normal(0,error)


def reverse_griewank_adjusted(x_adjusted):
    return -griewank_adjusted(x_adjusted)

# evaluated on x from [-500, 500], min of f is 0 at x = (420.9687, 420.9687)
def rastrigin(x):
    d = len(x)
    sumSin = 0
    for i in range(d):
        sumSin += x[i]**2 - 10*math.cos(2 * math.pi * x[i])

    return 10*d + sumSin

# accepts input in [0, 1]
def rastrigin_adjusted(x_adjusted, error=0):
    x = []
    for i in x_adjusted:
        x.append((i-.5)*5)
    return rastrigin(x) + np.random.normal(0, error)

# 1D function, made manually
# https://www.desmos.com/calculator/dfy1vkccwa
# accepts input in form [.5123]
def min3Parabola(x2):
    x1 = x2[0]

    p1 = 20*(x1-1/5)*(x1-1/3)-.3
    p2 = 50*(x1-1/3)*(x1-5/8)
    p3 = 15*(x1-1/2)*(x1-1)+.5
    # maxes = []
    # for i in range(len(p1)):
    #     maxes.append(max([p1[i], p2[i], p3[i]]))

    return min([p1, p2, p3])


def display1D(fun, x_range):
    x = np.linspace(x_range[0], x_range[1], 10000)
    y = [fun(i) for i in x]
    plt.plot(x, y)
    plt.show()

def display3D(fun, domain, ax, fColor='b', alpha=1, fineness = .005):

    x = y = np.arange(domain[0], domain[1], fineness)
    X, Y = np.meshgrid(x, y)
    zs = np.array([fun(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z, color=fColor, alpha=alpha)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # plt.show()