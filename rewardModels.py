# have the reward models take in instances and spit out reward/risk

# read params from a file for fit, restless, trad here
# will run when this file (rewardModels.py) is imported
# need discountFactor, windowLength
import kriging

discountFactor = .9
slidingWindow = 15

# outputs estimated value, variance
def fit(instance):
    # if len(instances.get_points()) > slidingWindow:

    points = instance.get_points()
    pointValues = instance.get_pointValues()
    if len(points) > slidingWindow:
        points = points[-slidingWindow:]
        pointValues = pointValues[-slidingWindow:]

    regResults = kriging.quadEstMin(points, pointValues, fitDiscount=discountFactor)

    # returns estimated minimum, variance
    return regResults[0], regResults[1]



def restless(instance):
    points = instance.get_points()
    pointValues = instance.get_pointValues()

    improvements = []
    for t in range(len(points)-1):
        improvement = pointValues[t+1] - pointValues[t]
        c_t = instance.get_gradDescentObject().get_ct(t)
        improvement /= c_t
        improvements.append(improvement)

    # finding weighted mean here
    if len(improvements) > slidingWindow:
        improvements = improvements[-slidingWindow:]

    denom = (1 - discountFactor ** len(improvements)) / (1 - discountFactor)
    empAverage = 0
    for i in range(len(improvements)):
        empAverage += improvements[len(improvements) - i - 1] * (discountFactor ** i)

    weightedMean = empAverage / denom

    variance = 0
    for i in range(len(improvements)):
        numer = (improvements[len(improvements) - i - 1] - weightedMean)**2
        numer *= (discountFactor ** i)
        variance += numer

    denom = (1 - discountFactor ** len(improvements)) / (1 - discountFactor)
    variance /= denom

    return weightedMean, variance



def trad(instance):
    pointValues = instance.get_pointValues()

    numer = 0
    avg = sum(pointValues) / len(pointValues)
    for fVal in pointValues:
        numer += (fVal - avg) ** 2
    variance = numer / (len(pointValues) + 1)

    return avg, variance


