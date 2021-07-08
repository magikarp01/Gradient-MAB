import math
import SPSA

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
    return -value


# ackley_adjusted accepts input with x_i in [0, 1], transforms from [-32.768, 32.768]
def ackley_adjusted():
    return 0