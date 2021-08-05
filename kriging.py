from scipy import optimize
import time



# discountRate <= 1
def fit_func(modelFunc, points, pointValues, numParams, discountRate = 1):
    d = len(points[0])
    def errorFunc(params):
        error = 0
        for pointIndex in range(len(points)):
            error += (pointValues[pointIndex] - modelFunc(points[pointIndex], params))**2
            error *= discountRate ** (len(points) - pointIndex - 1)
        return error


    x_0 = [1] * numParams
    leastSquares = optimize.minimize(errorFunc, x_0)
    optParams = list(leastSquares.x)
    # return error, function with optParams
    return errorFunc(optParams), lambda inp: modelFunc(inp, optParams)

# points is a set
def quadFit(points, pointValues):
    d = len(points[0])
    # params should have length d + (d choose 2) + d + 1
    # first d params are coefs of x_i^2
    # next (d choose 2) params are coefs of x_i*x_j
    # final is added
    def quadFunc(xPoint, params):
        val = params[-1]
        paramIndex = 0
        for i in range(d):
            val += params[i]*xPoint[i]**2
            paramIndex += 1
        for j in range(d):
            for k in range(j+1, d):
                val += params[paramIndex]*xPoint[j]*xPoint[k]
                paramIndex += 1
        for m in range(d):
            val += params[paramIndex]*xPoint[m]
            paramIndex += 1
        return val

    numParams = int(d + d*(d-1)/2 + d + 1)
    return fit_func(quadFunc, points, pointValues, numParams)


# probably a way to do this with linAlg
# https://calculus.subwiki.org/wiki/Quadratic_function_of_multiple_variables
def quadEstMin(points, pointValues):
    error, quadFunc = quadFit(points, pointValues)
    x_0 = [0]*len(points[0])
    result = optimize.minimize(quadFunc, x_0)
    # returns the variance, quadratic func
    return error, quadFunc(result.x)


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
# print(quadFit([(1,2), (2,8), (3, 3), (5, 6)], [6, 72, 27, 86]))
# print(quadFit([(1,), (2,), (3,)], [0, 2, 6]))
# print(quadEstMin([(1,2), (2,8), (3, 3), (5, 6)], [6, 72, 27, 86]))
# print("time is " + str(time.time() - startTime))