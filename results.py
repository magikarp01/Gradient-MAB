# Results are averaged over 1008 runs
# Parameters are in the tempOCBA function

import functions
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import multiprocessing as mp
import os

import gradientAllocation

plt.show()

import OCBAAlloc
import UCBAlloc
import uniformAlloc
import metaMaxAlloc
plt.show()

fun = functions.ackley_adjusted

def getAveFitOCBAError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                       iterations, startPosList,
                       minimum=0, discountRate=.8, a=.001, c=.001, useSPSA=False):
    errors = {}
    for iteration in tqdm(range(iterations)):
        results = OCBAAlloc.fitOCBASearch(fun, k, d, maxBudget, batchSize, numEvalsPerGrad,
                                          minSamples, discountRate=discountRate, a=a, c=c, useSPSA=useSPSA,
                                          startPos=startPosList[iteration])
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
                       iterations, startPosList, minimum=0, a=.001, c=.001, useSPSA=False):
    errors = {}
    for iteration in tqdm(range(iterations)):
        results = OCBAAlloc.tradOCBASearch(fun, k, d, maxBudget, batchSize, numEvalsPerGrad,
                                          minSamples, a=a, c=c, startPos = startPosList[iteration],
                                           useSPSA=useSPSA)
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
                      iterations, startPosList, minimum=0, discountRate=.8, a=.001, c=.001,useSPSA=False):
    errors = {}
    for iteration in tqdm(range(iterations)):
        results = UCBAlloc.fitUCBSearch(fun, k, d, maxBudget, batchSize, numEvalsPerGrad,
                                          minSamples, discountRate=discountRate, a=a, c=c,
                                        startPos = startPosList[iteration], useSPSA=useSPSA)
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
                      iterations, startPosList, minimum=0, a=.001, c=.001, startPos = False, useSPSA=False):
    errors = {}
    for iteration in tqdm(range(iterations)):
        results = UCBAlloc.tradUCBSearch(fun, k, d, maxBudget, batchSize, numEvalsPerGrad,
                                          minSamples, a=a, c=c, startPos = startPosList[iteration], useSPSA=useSPSA)
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
                    iterations, startPosList, minimum=0, a=.001, c=.001, startPos = False, useSPSA=False):
    errors = {}
    for iteration in tqdm(range(iterations)):
        results = uniformAlloc.uniformSearch(fun, k, d, maxBudget, batchSize, numEvalsPerGrad,
                        a=a, c=c, startPos = startPosList[iteration], useSPSA=useSPSA)
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
                    iterations, startPosList, minimum=0, a=.001, c=.001, startPos = False, useSPSA=False):
    errors = {}
    for iteration in tqdm(range(iterations)):
        results = metaMaxAlloc.metaMaxSearch(fun, k, d, maxBudget, batchSize, numEvalsPerGrad,
                        a=a, c=c, startPos = startPosList[iteration], useSPSA=useSPSA)
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
# useSPSA = False

# errors, aveError = getAveOCBAError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
#                            iterations, discountRate=discountRate, a=a, c=c, useSPSA=useSPSA)

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
    useSPSA = sharedParams[11]
    startPosList = sharedParams[12]

    errors, aveError = getAveFitOCBAError(fun, k, d, maxBudget, batchSize,
                                          numEvalsPerGrad, minSamples, iterations, startPosList,
                                          minimum=minimum, discountRate=discountRate, a=a, c=c, useSPSA=useSPSA)
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
    useSPSA = sharedParams[11]
    startPosList = sharedParams[12]
    errors, aveError = getAveTradOCBAError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                           iterations, startPosList,
                                          minimum=minimum, a=a, c=c, useSPSA=useSPSA)
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
    useSPSA = sharedParams[11]
    startPosList = sharedParams[12]
    errors, aveError = getAveFitUCBError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                         iterations, startPosList,
                                         minimum=minimum, discountRate=discountRate, a=a, c=c, useSPSA=useSPSA)
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
    useSPSA = sharedParams[11]
    startPosList = sharedParams[12]
    errors, aveError = getAveTradUCBError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                          iterations, startPosList,
                                         minimum=minimum, a=a, c=c, useSPSA=useSPSA)
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
    useSPSA = sharedParams[11]
    startPosList = sharedParams[12]
    errors, aveError = getAveUniformError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad,
                                          iterations, startPosList,
                                       minimum=minimum, a=a, c=c, useSPSA=useSPSA)

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
    useSPSA = sharedParams[11]
    startPosList = sharedParams[12]
    errors, aveError = getAveMetaMaxError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad,
                                          iterations, startPosList,
                                       minimum=minimum, a=a, c=c, useSPSA=useSPSA)

    aveErrorList.append(aveError)



def multiprocessSearch(numProcesses, iterations, func, sharedParams, endPath):
    if __name__ == '__main__':
        with mp.Manager() as manager:
            aveErrorList = manager.list()
            print("ID of main process: {}".format(os.getpid()))
            processes = []
            for i in range(numProcesses):
                startPosList = processStartPos[i]
                processes.append(mp.Process(target=func, args=(aveErrorList, iterations, sharedParams, startPosList)))
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


def performMultiprocess(numProcesses, iterPerProcess):
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
    useSPSA = False

    processStartPos = []
    for i in range(numProcesses):
        startPosList = []
        for j in range(iterPerProcess):
            startPosList.append(gradientAllocation.stratifiedSampling(d, k))
        processStartPos.append(startPosList)

    sharedParams = [fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                    minimum, discountRate, a, c, useSPSA, processStartPos]

    if __name__ == '__main__':
        dir = "Results\\averageErrors\\"
        # multiprocessSearch(numProcesses, iterPerProcess, tempFitOCBA, sharedParams, dir+"fitOCBA.json")

        # multiprocessSearch(numProcesses, iterPerProcess, tempTradOCBA, sharedParams, dir + "tradOCBA.json")

        # multiprocessSearch(numProcesses, iterPerProcess, tempFitUCB, sharedParams, dir + "fitUCB.json")

        # multiprocessSearch(numProcesses, iterPerProcess, tempTradUCB, sharedParams, dir + "tradUCB.json")

        multiprocessSearch(numProcesses, iterPerProcess, tempMetaMax, sharedParams, dir + "metaMax.json")

        multiprocessSearch(numProcesses, iterPerProcess, tempUniform, sharedParams, dir + "uniform.json")


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

performMultiprocess(2, 10)

# plt.clf()

# path = "Results\\averageErrors"
# fileNames = os.listdir(path)
# names = [fileName[:-5] for fileName in fileNames]
# dics = []
# for fileName in fileNames:
#     with open(path + "\\" + fileName) as jf:
#         dics.append(json.load(jf))
# showMinimaHistory(dics, names)
