import gradientAllocation
import tests
import functions
import fitAlloc
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import multiprocessing
import os
import uniformAlloc
import metaMaxAlloc

fun = functions.ackley_adjusted

def getAveOCBAError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                    iterations, minimum=0, discountRate=.8, a=.001, c=.001, startPos = False):
    errors = {}
    for iteration in tqdm(range(iterations)):
        results = fitAlloc.OCBASearch(fun, k, d, maxBudget, batchSize, numEvalsPerGrad,
                        minSamples, discountRate=discountRate, a=a, c=c, startPos = startPos)
        # ideally, every convergeDic has the same keys
        convergeDic = results[2]
        for s in convergeDic.keys():
            try:
                errors[s].append( convergeDic[s] - minimum)
            except:
                errors[s] = [convergeDic[s] - minimum]

    aveError = {}
    for s in errors.keys():
        aveError[s] = float(sum(errors[s])) / len(errors[s])

    return errors, aveError


def getAveUniformError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad,
                    iterations, minimum=0, a=.001, c=.001, startPos = False):
    errors = {}
    for iteration in tqdm(range(iterations)):
        results = uniformAlloc.uniformSearch(fun, k, d, maxBudget, batchSize, numEvalsPerGrad,
                        a=a, c=c, startPos = startPos)
        # ideally, every convergeDic has the same keys
        convergeDic = results[2]
        for s in convergeDic.keys():
            try:
                errors[s].append( convergeDic[s] - minimum)
            except:
                errors[s] = [convergeDic[s] - minimum]

    aveError = {}
    for s in errors.keys():
        aveError[s] = float(sum(errors[s])) / len(errors[s])

    return errors, aveError


def getAveMetaMaxError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad,
                    iterations, minimum=0, a=.001, c=.001, startPos = False):
    errors = {}
    for iteration in tqdm(range(iterations)):
        results = metaMaxAlloc.metaMaxSearch(fun, k, d, maxBudget, batchSize, numEvalsPerGrad,
                        a=a, c=c, startPos = startPos)
        # ideally, every convergeDic has the same keys
        convergeDic = results[2]
        for s in convergeDic.keys():
            try:
                errors[s].append( convergeDic[s] - minimum)
            except:
                errors[s] = [convergeDic[s] - minimum]

    aveError = {}
    for s in errors.keys():
        aveError[s] = float(sum(errors[s])) / len(errors[s])

    return errors, aveError
# fun = functions.ackley_adjusted
# k = 5
# d = 2
# maxBudget = 10000
# batchSize = 100
# numEvalsPerGrad = 2*d
# minSamples = 10
#
# iterations = 5
#
# minimum = 0
# discountRate = .8
# a = .002
# c = .000001

# errors, aveError = getAveOCBAError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
#                            iterations, discountRate=discountRate, a=a, c=c)

def tempOCBA(jsonNum):
    fun = functions.ackley_adjusted
    k = 5
    d = 2
    maxBudget = 10000
    batchSize = 100
    numEvalsPerGrad = 2 * d
    minSamples = 10

    iterations = 63

    minimum = 0
    discountRate = .8
    a = .002
    c = .000001
    errors, aveError = getAveOCBAError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples, iterations,
                                       minimum=minimum, discountRate=discountRate, a=a, c=c)
    with open('errors' + str(jsonNum) + '.json', 'w') as fp:
        json.dump(errors, fp)


    with open('aveError' + str(jsonNum) + '.json', 'w') as fp:
        json.dump(aveError, fp)


def tempUniform(jsonNum):
    fun = functions.ackley_adjusted
    k = 5
    d = 2
    maxBudget = 10000
    batchSize = 100
    numEvalsPerGrad = 2 * d
    iterations = 63

    minimum = 0
    a = .002
    c = .000001
    errors, aveError = getAveUniformError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, iterations,
                                       minimum=minimum, a=a, c=c)
    with open('errors' + str(jsonNum) + '.json', 'w') as fp:
        json.dump(errors, fp)

    with open('aveError' + str(jsonNum) + '.json', 'w') as fp:
        json.dump(aveError, fp)


def tempMetaMax(jsonNum):
    fun = functions.ackley_adjusted
    k = 5
    d = 2
    maxBudget = 10000
    batchSize = 100
    numEvalsPerGrad = 2 * d
    iterations = 63

    minimum = 0
    a = .002
    c = .000001
    errors, aveError = getAveMetaMaxError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, iterations,
                                       minimum=minimum, a=a, c=c)
    with open('errors' + str(jsonNum) + '.json', 'w') as fp:
        json.dump(errors, fp)

    with open('aveError' + str(jsonNum) + '.json', 'w') as fp:
        json.dump(aveError, fp)



def multiprocessSearch(func):
    if __name__ == '__main__':
        print("ID of main process: {}".format(os.getpid()))
        p1 = multiprocessing.Process(target=func, args=(1,))
        p2 = multiprocessing.Process(target=func, args=(2,))
        p3 = multiprocessing.Process(target=func, args=(3,))
        p4 = multiprocessing.Process(target=func, args=(4,))
        p5 = multiprocessing.Process(target=func, args=(5,))
        p6 = multiprocessing.Process(target=func, args=(6,))
        p7 = multiprocessing.Process(target=func, args=(7,))
        p8 = multiprocessing.Process(target=func, args=(8,))
        p9 = multiprocessing.Process(target=func, args=(9,))
        p10 = multiprocessing.Process(target=func, args=(10,))
        p11 = multiprocessing.Process(target=func, args=(11,))
        p12 = multiprocessing.Process(target=func, args=(12,))
        p13 = multiprocessing.Process(target=func, args=(13,))
        p14 = multiprocessing.Process(target=func, args=(14,))
        p15 = multiprocessing.Process(target=func, args=(15,))
        p16 = multiprocessing.Process(target=func, args=(16,))

        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        p6.start()
        p7.start()
        p8.start()
        p9.start()
        p10.start()
        p11.start()
        p12.start()
        p13.start()
        p14.start()
        p15.start()
        p16.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()
        p6.join()
        p7.join()
        p8.join()
        p9.join()
        p10.join()
        p11.join()
        p12.join()
        p13.join()
        p14.join()
        p15.join()
        p16.join()

        print("All processes finished execution!")

        # check if processes are alive
        print('Process p1 is alive: {}'.format(p1.is_alive()))
        print('Process p2 is alive: {}'.format(p2.is_alive()))
        print('Process p3 is alive: {}'.format(p3.is_alive()))
        print('Process p4 is alive: {}'.format(p4.is_alive()))
        print('Process p5 is alive: {}'.format(p5.is_alive()))
        print('Process p6 is alive: {}'.format(p6.is_alive()))
        print('Process p7 is alive: {}'.format(p7.is_alive()))
        print('Process p8 is alive: {}'.format(p8.is_alive()))
        print('Process p9 is alive: {}'.format(p9.is_alive()))
        print('Process p10 is alive: {}'.format(p10.is_alive()))
        print('Process p11 is alive: {}'.format(p11.is_alive()))
        print('Process p12 is alive: {}'.format(p12.is_alive()))
        print('Process p13 is alive: {}'.format(p13.is_alive()))
        print('Process p14 is alive: {}'.format(p14.is_alive()))
        print('Process p15 is alive: {}'.format(p15.is_alive()))
        print('Process p16 is alive: {}'.format(p16.is_alive()))


def storeAverageError(name):
    aveError = {}
    for i in range(1, 17):
        with open('aveError' + str(i) + '.json') as jf:
            subAveDic = json.load(jf)
            for sampleNum in subAveDic.keys():
                try:
                    aveError[sampleNum] += subAveDic[sampleNum]
                except:
                    aveError[sampleNum] = subAveDic[sampleNum]

    for sampleNum in aveError.keys():
        aveError[sampleNum] = aveError[sampleNum] / 16

    with open(name, 'w') as fp:
        json.dump(aveError, fp)

def storeErrors(name):
    errors = {}
    for i in range(1, 17):
        with open('errors' + str(i) + '.json') as jf:
            subAveDic = json.load(jf)
            for sampleNum in subAveDic.keys():
                try:
                    errors[sampleNum] += subAveDic[sampleNum]
                except:
                    errors[sampleNum] = [subAveDic[sampleNum]]

    with open(name, 'w') as fp:
        json.dump(errors, fp)

# multiprocessSearch(tempMetaMax)
# storeAverageError("uniformAverageError.json")
# storeErrors("uniformErrors.json")

def showMinimaHistory():
    with open('OCBAAverageError.json') as jf:
        OCBAAveError = json.load(jf)

    with open('uniformAverageError.json') as jf:
        uniformAveError = json.load(jf)

    with open('metaMaxAverageError.json') as jf:
        metaMaxAveError = json.load(jf)

    fig, ax = plt.subplots(1)
    ax.title.set_text("Average Error History")
    ax.set_xlabel("Total Samples")
    ax.set_ylabel("Error")

    x1 = list(OCBAAveError.keys())
    y1 = [OCBAAveError[m] for m in x1]

    x2 = list(uniformAveError.keys())
    y2 = [uniformAveError[m] for m in x2]

    x3 = list(metaMaxAveError.keys())
    y3 = [metaMaxAveError[m] for m in x3]

    x1 = [int(i) for i in x1]
    x2 = [int(i) for i in x1]
    x3 = [int(i) for i in x1]

    ax.plot(x1, y1, label="OCBA")
    ax.plot(x2, y2, label="Uniform")
    ax.plot(x3, y3, label="MetaMax")

    ax.legend(loc="upper right")

    plt.show()

showMinimaHistory()
