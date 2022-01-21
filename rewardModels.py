# have the reward models take in instances and spit out reward/risk

# read params from a file for fit, restless, trad here
# will run when this file (rewardModels.py) is imported
# need discountFactor, windowLength
discountFactor = .9
slidingWindow = 15

# outputs variance and estimated value
def fit(instance):
    # if len(instances.get_points()) > slidingWindow:

    d = len(instance.get_lastPoint()[0])

    points = instance

    regResults = kriging.quad

# def restless(instances):


# def trad(instances):