from scipy import optimize
import time



# fitDiscount <= 1

# max number of points should be sliding window
def fit_func(modelFunc, points, pointValues, numParams, paramBounds, fitDiscount = .9):

    d = len(points[0])

    if (len(points) <= 2 *d + 2):
        raise Exception("Number of points is too small!")

    # stop considering points when fitDiscount < stopConsidering
    def errorFunc(params):
        error = 0
        totalDiscount = 1
        for pointIndex in range(len(points)-1, -1, -1):
            dif = pointValues[pointIndex] - modelFunc(points[pointIndex], params)
            error += dif**2 * totalDiscount
            totalDiscount*=fitDiscount
        return error

    x_0 = [1] * numParams
    leastSquares = optimize.minimize(errorFunc, x_0, bounds=paramBounds, options={'maxiter':60})
    optParams = list(leastSquares.x)
    # return error, function with optParams
    # return errorFunc(optParams), lambda inp: modelFunc(inp, optParams)
    return optParams, errorFunc(optParams)

# points is a set
def parabFit(points, pointValues):
    d = len(points[0])
    # params should have length 2d + 1
    # first d params are coefs of x_i^2
    # next d params are coefs of x_i
    # final is added
    # minimum is at xPoint = [params[d], ..., params[2*d-1]]
    # minimum = params[2*d]
    def parabFunc(xPoint, params):
        val = params[-1]
        for i in range(d):
            val += params[i]*(xPoint[i] - params[i+d])**2
        return val

    numParams = 2*d + 1
    aBounds = [(0, None) for i in range(d)]
    otherBounds = [(None, None) for i in range(d+1)]
    paramBounds = aBounds + otherBounds
    return fit_func(parabFunc, points, pointValues, numParams, paramBounds)


# probably a way to do this with linAlg
# https://calculus.subwiki.org/wiki/Quadratic_function_of_multiple_variables
def quadEstMin(points, pointValues, fitDiscount = .9):
    optParams, error = parabFit(points, pointValues)

    result = optParams[-1]
    n = len(points)

    # denom is for error to variance conversion
    if fitDiscount==1:
        denom = n
    else:
        denom = (1 - fitDiscount**n)/(1 - fitDiscount)

    return result, error/denom


def linFit(points, pointValues):
    d = len(points[0])
    # d+1 params
    def linFunc(xPoint, params):
        val = params[-1]
        for i in range(d):
            val += params[i]*xPoint[i]
        return val

    return fit_func(linFunc, points, pointValues)

def expFit(points, pointValues):
    return

# startTime = time.time()
# print(parabFit([(1,2), (2,8), (3, 3), (5, 6)], [6, 72, 27, 86]))
# print(parabFit([(1,), (2,), (3,)], [0, 2, 6]))
# print(quadEstMin([(1,2), (2,8), (3, 3), (5, 6)], [6, 72, 27, 86]))
# print("time is " + str(time.time() - startTime))