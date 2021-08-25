# Results are averaged over 1008 runs
# Parameters are in the tempOCBA function

import functions
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import multiprocessing
import os

plt.show()

import OCBAAlloc
import uniformAlloc
import metaMaxAlloc
plt.show()

fun = functions.ackley_adjusted

def getAveOCBAError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                    iterations, minimum=0, discountRate=.8, a=.001, c=.001, startPos = False):
    errors = {}
    for iteration in tqdm(range(iterations)):
        results = OCBAAlloc.OCBASearch(fun, k, d, maxBudget, batchSize, numEvalsPerGrad,
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

def tempOCBA(jsonNum, path):
    fun = functions.griewank_adjusted
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
    with open(path + '\\errors' + str(jsonNum) + '.json', 'w') as fp:
        json.dump(errors, fp)

    with open(path + '\\aveError' + str(jsonNum) + '.json', 'w') as fp:
        json.dump(aveError, fp)


def tempUniform(jsonNum, path):
    fun = functions.griewank_adjusted
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
    with open(path + '\\errors' + str(jsonNum) + '.json', 'w') as fp:
        json.dump(errors, fp)

    with open(path + '\\aveError' + str(jsonNum) + '.json', 'w') as fp:
        json.dump(aveError, fp)


def tempMetaMax(jsonNum, path):
    fun = functions.griewank_adjusted
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
    with open(path + '\\errors' + str(jsonNum) + '.json', 'w') as fp:
        json.dump(errors, fp)

    with open(path + '\\aveError' + str(jsonNum) + '.json', 'w') as fp:
        json.dump(aveError, fp)



def multiprocessSearch(func, path):
    if __name__ == '__main__':
        print("ID of main process: {}".format(os.getpid()))
        p1 = multiprocessing.Process(target=func, args=(1, path, ))
        p2 = multiprocessing.Process(target=func, args=(2, path, ))
        p3 = multiprocessing.Process(target=func, args=(3, path, ))
        p4 = multiprocessing.Process(target=func, args=(4, path, ))
        p5 = multiprocessing.Process(target=func, args=(5, path, ))
        p6 = multiprocessing.Process(target=func, args=(6, path, ))
        p7 = multiprocessing.Process(target=func, args=(7, path, ))
        p8 = multiprocessing.Process(target=func, args=(8, path, ))
        p9 = multiprocessing.Process(target=func, args=(9, path, ))
        p10 = multiprocessing.Process(target=func, args=(10, path, ))
        p11 = multiprocessing.Process(target=func, args=(11, path, ))
        p12 = multiprocessing.Process(target=func, args=(12, path, ))
        p13 = multiprocessing.Process(target=func, args=(13, path, ))
        p14 = multiprocessing.Process(target=func, args=(14, path, ))
        p15 = multiprocessing.Process(target=func, args=(15, path, ))
        p16 = multiprocessing.Process(target=func, args=(16, path, ))

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


def storeAverageError(name, path):
    aveError = {}
    for i in range(1, 17):
        with open(path + '\\aveError' + str(i) + '.json') as jf:
            subAveDic = json.load(jf)
            for sampleNum in subAveDic.keys():
                try:
                    aveError[sampleNum] += subAveDic[sampleNum]
                except:
                    aveError[sampleNum] = subAveDic[sampleNum]

    for sampleNum in aveError.keys():
        aveError[sampleNum] = aveError[sampleNum] / 16

    with open(path + "\\" + name, 'w') as fp:
        json.dump(aveError, fp)

def storeErrors(name, path):
    errors = {}
    for i in range(1, 17):
        with open(path + '\\errors' + str(i) + '.json') as jf:
            subAveDic = json.load(jf)
            for sampleNum in subAveDic.keys():
                try:
                    errors[sampleNum] += subAveDic[sampleNum]
                except:
                    errors[sampleNum] = [subAveDic[sampleNum]]

    with open(path + "\\" + name, 'w') as fp:
        json.dump(errors, fp)


path = "Results\\metaMaxErrorFiles"
# multiprocessSearch(tempOCBA, path)



# path = "Results\\metaMaxErrorFiles"
# storeAverageError("metaMaxAverageError.json", path)
# storeErrors("metaMaxErrors.json", path)
#
# path = "Results\\uniformErrorFiles"
# storeAverageError("uniformAverageError.json", path)
# storeErrors("uniformErrors.json", path)
#
# path = "Results\\OCBAErrorFiles"
# storeAverageError("OCBAAverageError.json", path)
# storeErrors("OCBAErrors.json", path)

def showMinimaHistory(path):
    with open(path + '\\OCBAAverageError.json') as jf:
        OCBAAveError = json.load(jf)

    with open(path + '\\uniformAverageError.json') as jf:
        uniformAveError = json.load(jf)

    with open(path + '\\metaMaxAverageError.json') as jf:
        metaMaxAveError = json.load(jf)


    x1 = list(OCBAAveError.keys())
    y1 = [OCBAAveError[m] for m in x1]

    x2 = list(uniformAveError.keys())
    y2 = [uniformAveError[m] for m in x2]

    x3 = list(metaMaxAveError.keys())
    y3 = [metaMaxAveError[m] for m in x3]

    x1 = [int(i) for i in x1]
    x2 = [int(i) for i in x1]
    x3 = [int(i) for i in x1]

    fig, ax = plt.subplots(1)

    ax.plot(x1, y1, label="OCBA")
    ax.plot(x2, y2, label="Uniform")
    ax.plot(x3, y3, label="MetaMax")

    ax.title.set_text("Average Error History")
    ax.set_xlabel("Total Samples")
    ax.set_ylabel("Error")

    ax.legend(loc="upper right")
    plt.semilogx()
    plt.show()

plt.clf()
showMinimaHistory("Results\\2D Griewank Results")

