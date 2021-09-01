# Results are averaged over 1008 runs
# Parameters are in the tempOCBA function

import functions
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import multiprocessing as mp
import os

plt.show()

import OCBAAlloc
import UCBAlloc
import uniformAlloc
import metaMaxAlloc
plt.show()

fun = functions.ackley_adjusted

def getAveFitOCBAError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                       iterations, minimum=0, discountRate=.8, a=.001, c=.001, startPos = False):
    errors = {}
    for iteration in tqdm(range(iterations)):
        results = OCBAAlloc.fitOCBASearch(fun, k, d, maxBudget, batchSize, numEvalsPerGrad,
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

def getAveTradOCBAError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                       iterations, minimum=0, a=.001, c=.001, startPos = False):
    errors = {}
    for iteration in tqdm(range(iterations)):
        results = OCBAAlloc.tradOCBASearch(fun, k, d, maxBudget, batchSize, numEvalsPerGrad,
                                          minSamples, a=a, c=c, startPos = startPos)
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

def getAveFitUCBError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                      iterations, minimum=0, discountRate=.8, a=.001, c=.001, startPos = False):
    errors = {}
    for iteration in tqdm(range(iterations)):
        results = UCBAlloc.fitUCBSearch(fun, k, d, maxBudget, batchSize, numEvalsPerGrad,
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

def getAveTradUCBError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                      iterations, minimum=0, a=.001, c=.001, startPos = False):
    errors = {}
    for iteration in tqdm(range(iterations)):
        results = UCBAlloc.tradUCBSearch(fun, k, d, maxBudget, batchSize, numEvalsPerGrad,
                                          minSamples, a=a, c=c, startPos = startPos)
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

def tempFitOCBA(aveErrorList, iterations, sharedParams):
    fun = functions.griewank_adjusted
    k = 5
    d = 2
    maxBudget = 5000
    batchSize = 100
    numEvalsPerGrad = 2 * d
    minSamples = 10

    minimum = 0
    discountRate = .8
    a = .002
    c = .000001
    errors, aveError = getAveFitOCBAError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples, iterations,
                                          minimum=minimum, discountRate=discountRate, a=a, c=c)
    aveErrorList.append(aveError)


def tempTradOCBA(aveErrorList, iterations):
    fun = functions.griewank_adjusted
    k = 5
    d = 2
    maxBudget = 5000
    batchSize = 100
    numEvalsPerGrad = 2 * d
    minSamples = 10

    minimum = 0
    a = .002
    c = .000001
    errors, aveError = getAveTradOCBAError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples, iterations,
                                          minimum=minimum, a=a, c=c)
    aveErrorList.append(aveError)


def tempFitUCB(aveErrorList, iterations):
    fun = functions.griewank_adjusted
    k = 5
    d = 2
    maxBudget = 10000
    batchSize = 100
    numEvalsPerGrad = 2 * d
    minSamples = 10

    minimum = 0
    discountRate = .8
    a = .002
    c = .000001
    errors, aveError = getAveFitUCBError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples, iterations,
                                         minimum=minimum, discountRate=discountRate, a=a, c=c)
    aveErrorList.append(aveError)


def tempUniform(aveErrorList, iterations):
    fun = functions.griewank_adjusted
    k = 5
    d = 2
    maxBudget = 10000
    batchSize = 100
    numEvalsPerGrad = 2 * d

    minimum = 0
    a = .002
    c = .000001
    errors, aveError = getAveUniformError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, iterations,
                                       minimum=minimum, a=a, c=c)

    aveErrorList.append(aveError)


def tempMetaMax(aveErrorList, iterations):
    fun = functions.griewank_adjusted
    k = 5
    d = 2
    maxBudget = 10000
    batchSize = 100
    numEvalsPerGrad = 2 * d

    minimum = 0
    a = .002
    c = .000001
    errors, aveError = getAveMetaMaxError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, iterations,
                                       minimum=minimum, a=a, c=c)

    aveErrorList.append(aveError)



def multiprocessSearch(numProcesses, iterations, func, endPath):
    if __name__ == '__main__':
        with mp.Manager() as manager:
            aveErrorList = manager.list()
            print("ID of main process: {}".format(os.getpid()))
            processes = []
            for i in range(numProcesses):
                processes.append(mp.Process(target=func, args=(aveErrorList, iterations)))
                processes[i].start()

            for i in range(numProcesses):
                processes[i].join()

            print("All processes finished execution!")

            # check if processes are alive
            for i in range(numProcesses):
                print('Process p' + str(i+1) + ' is alive: {}'.format(processes[i].is_alive()))

            aveError = {}
            for i in range(numProcesses):
                for sampleNum in aveErrorList[i].keys():
                    try:
                        aveError[sampleNum] += aveErrorList[i][sampleNum]
                    except:
                        aveError[sampleNum] = aveErrorList[i][sampleNum]

            for sampleNum in aveError.keys():
                aveError[sampleNum] = aveError[sampleNum] / numProcesses

            with open(endPath, 'w') as fp:
                json.dump(aveError, fp)

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



fun = functions.griewank_adjusted
k = 5
d = 2
maxBudget = 10000
batchSize = 20
numEvalsPerGrad = 2 * d
minSamples = 10

minimum = 0
discountRate = .8
a = .002
c = .000001
sharedParams = [fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples, minimum, discountRate, a, c]

if __name__ == '__main__':
    path = "Results\\OCBAAverageError.json"
    multiprocessSearch(4, 3, tempFitOCBA, path)

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
    #
    # path = "Results\\UCBErrorFiles"
    # storeAverageError("UCBAverageError.json", path)
    # storeErrors("UCBErrors.json", path)


def showMinimaHistory(path):
    with open(path + '\\OCBAAverageError.json') as jf:
        OCBAAveError = json.load(jf)

    # with open(path + '\\UCBAverageError.json') as jf:
    #     UCBAveError = json.load(jf)

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

    # x4 = list(UCBAveError.keys())
    # y4 = [UCBAveError[m] for m in x4]

    x1 = [int(i) for i in x1]
    x2 = [int(i) for i in x2]
    x3 = [int(i) for i in x3]
    # x4 = [int(i) for i in x4]

    fig, ax = plt.subplots(1)

    ax.plot(x1, y1, label="OCBA")
    ax.plot(x2, y2, label="Uniform")
    ax.plot(x3, y3, label="MetaMax")
    # ax.plot(x4, y4, label="UCB")

    ax.title.set_text("Average Error History")
    ax.set_xlabel("Total Samples")
    ax.set_ylabel("Error")

    ax.legend(loc="upper right")
    # plt.semilogx()
    plt.show()

# plt.clf()
# showMinimaHistory("Results\\2D Ackley Results")

