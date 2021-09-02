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
    fun = sharedParams[0]
    k = sharedParams[1]
    d = sharedParams[2]
    maxBudget = sharedParams[3]
    batchSize = sharedParams[4]
    numEvalsPerGrad = sharedParams[5]
    minSamples = sharedParams[6]

    minimum = sharedParams[7]
    discountRate = sharedParams[8]
    a = sharedParams[9]
    c = sharedParams[10]
    errors, aveError = getAveFitOCBAError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples, iterations,
                                          minimum=minimum, discountRate=discountRate, a=a, c=c)
    aveErrorList.append(aveError)


def tempTradOCBA(aveErrorList, iterations, sharedParams):
    fun = sharedParams[0]
    k = sharedParams[1]
    d = sharedParams[2]
    maxBudget = sharedParams[3]
    batchSize = sharedParams[4]
    numEvalsPerGrad = sharedParams[5]
    minSamples = sharedParams[6]

    minimum = sharedParams[7]
    a = sharedParams[9]
    c = sharedParams[10]
    errors, aveError = getAveTradOCBAError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples, iterations,
                                          minimum=minimum, a=a, c=c)
    aveErrorList.append(aveError)


def tempFitUCB(aveErrorList, iterations, sharedParams):
    fun = sharedParams[0]
    k = sharedParams[1]
    d = sharedParams[2]
    maxBudget = sharedParams[3]
    batchSize = sharedParams[4]
    numEvalsPerGrad = sharedParams[5]
    minSamples = sharedParams[6]

    minimum = sharedParams[7]
    discountRate = sharedParams[8]
    a = sharedParams[9]
    c = sharedParams[10]
    errors, aveError = getAveFitUCBError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples, iterations,
                                         minimum=minimum, discountRate=discountRate, a=a, c=c)
    aveErrorList.append(aveError)


def tempTradUCB(aveErrorList, iterations, sharedParams):
    fun = sharedParams[0]
    k = sharedParams[1]
    d = sharedParams[2]
    maxBudget = sharedParams[3]
    batchSize = sharedParams[4]
    numEvalsPerGrad = sharedParams[5]
    minSamples = sharedParams[6]

    minimum = sharedParams[7]
    a = sharedParams[9]
    c = sharedParams[10]
    errors, aveError = getAveTradUCBError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples, iterations,
                                         minimum=minimum, a=a, c=c)
    aveErrorList.append(aveError)


def tempUniform(aveErrorList, iterations, sharedParams):
    fun = sharedParams[0]
    k = sharedParams[1]
    d = sharedParams[2]
    maxBudget = sharedParams[3]
    batchSize = sharedParams[4]
    numEvalsPerGrad = sharedParams[5]
    minSamples = sharedParams[6]

    minimum = sharedParams[7]
    a = sharedParams[9]
    c = sharedParams[10]
    errors, aveError = getAveUniformError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, iterations,
                                       minimum=minimum, a=a, c=c)

    aveErrorList.append(aveError)


def tempMetaMax(aveErrorList, iterations, sharedParams):
    fun = sharedParams[0]
    k = sharedParams[1]
    d = sharedParams[2]
    maxBudget = sharedParams[3]
    batchSize = sharedParams[4]
    numEvalsPerGrad = sharedParams[5]

    minimum = sharedParams[7]

    a = sharedParams[9]
    c = sharedParams[10]
    errors, aveError = getAveMetaMaxError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, iterations,
                                       minimum=minimum, a=a, c=c)

    aveErrorList.append(aveError)



def multiprocessSearch(numProcesses, iterations, func, sharedParams, endPath):
    if __name__ == '__main__':
        with mp.Manager() as manager:
            aveErrorList = manager.list()
            print("ID of main process: {}".format(os.getpid()))
            processes = []
            for i in range(numProcesses):
                processes.append(mp.Process(target=func, args=(aveErrorList, iterations, sharedParams)))
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


def performMultiprocess():
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
    sharedParams = [fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples, minimum, discountRate, a, c]

    if __name__ == '__main__':
        dir = "Results\\averageErrors\\"
        # multiprocessSearch(16, 63, tempFitOCBA, sharedParams, dir+"fitOCBA.json")

        multiprocessSearch(16, 63, tempTradOCBA, sharedParams, dir + "tradOCBA.json")

        # multiprocessSearch(16, 63, tempFitUCB, sharedParams, dir + "fitUCB.json")

        multiprocessSearch(16, 63, tempTradUCB, sharedParams, dir + "tradUCB.json")

        multiprocessSearch(16, 63, tempMetaMax, sharedParams, dir + "metaMax.json")

        multiprocessSearch(16, 63, tempUniform, sharedParams, dir + "uniform.json")

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



def showMinimaHistory(dics, names):
    fig, ax = plt.subplots(1)

    for i in range(len(dics)):
        # jf = files[i]
        name = names[i]
        # aveError = json.load(jf)
        aveError = dics[i]
        x = list(aveError.keys())
        y = [aveError[m] for m in x]
        x = [int(i) for i in x]
        ax.plot(x, y, label=name)

    ax.title.set_text("Average Error History")
    ax.set_xlabel("Total Samples")
    ax.set_ylabel("Error")

    ax.legend(loc="upper right")
    # plt.semilogx()
    plt.show()



# plt.clf()

path = "Results\\averageErrors"
fileNames = os.listdir(path)
names = [fileName[:-5] for fileName in fileNames]
dics = []
for fileName in fileNames:
    with open(path + "\\" + fileName) as jf:
        dics.append(json.load(jf))
showMinimaHistory(dics, names)
