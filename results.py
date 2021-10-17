# Results are averaged over 1008 runs
# Parameters are in the tempOCBA function

import functions
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import multiprocessing as mp
import os

import gradientAllocation
import paramPickler
import baiAllocations
import fitBandits
import restlessBandits
import tradBandits
import uniformAlloc
import metaMaxAlloc

import sys


fun = functions.ackley_adjusted

def tempFitOCBA(aveErrorList, iterations, sharedParams, startPosList):
    f = sharedParams[0]
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

    errors = {}
    for iteration in tqdm(range(iterations)):
        results = fitBandits.fitInfiniteSearch(baiAllocations.OCBA.getBudget, f, d, maxBudget, batchSize,
                                               numEvalsPerGrad, minSamples, discountRate=discountRate,
                                               a=a, c=c, useTqdm=False, useSPSA=useSPSA)
        convergeDic = results[2]
        for s in convergeDic.keys():
            try:
                errors[s].append(convergeDic[s] - minimum)
            except:
                errors[s] = [convergeDic[s] - minimum]

    aveError = {}
    for s in errors.keys():
        aveError[s] = float(sum(errors[s])) / len(errors[s])

    aveErrorList.append(aveError)

def tempFitInfiniteOCBA(aveErrorList, iterations, sharedParams, startPosList):
    f = sharedParams[0]
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

    errors = {}
    for iteration in tqdm(range(iterations)):
        results = fitBandits.fitSearch(baiAllocations.OCBA.getBudget, f, d, maxBudget, batchSize, numEvalsPerGrad,
                                       minSamples, discountRate=discountRate,
                                       a=a, c=c, useTqdm=False, useSPSA=useSPSA)
        # ideally, every convergeDic has the same keys
        convergeDic = results[2]
        for s in convergeDic.keys():
            try:
                errors[s].append(convergeDic[s] - minimum)
            except:
                errors[s] = [convergeDic[s] - minimum]

    aveError = {}
    for s in errors.keys():
        aveError[s] = float(sum(errors[s])) / len(errors[s])

    aveErrorList.append(aveError)

def tempFitUCB(aveErrorList, iterations, sharedParams, startPosList):
    f = sharedParams[0]
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

    errors = {}
    for iteration in tqdm(range(iterations)):
        results = fitBandits.fitSearch(baiAllocations.UCB.getBudget, f, k, d, maxBudget, batchSize, numEvalsPerGrad,
                                       minSamples, discountRate=discountRate,
                                       a=a, c=c, startPos=startPosList[iteration], useTqdm=False, useSPSA=useSPSA)
        # ideally, every convergeDic has the same keys
        convergeDic = results[2]
        for s in convergeDic.keys():
            try:
                errors[s].append(convergeDic[s] - minimum)
            except:
                errors[s] = [convergeDic[s] - minimum]

    aveError = {}
    for s in errors.keys():
        aveError[s] = float(sum(errors[s])) / len(errors[s])

    aveErrorList.append(aveError)

def tempFitInfiniteUCB(aveErrorList, iterations, sharedParams, startPosList):
    f = sharedParams[0]
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

    errors = {}
    for iteration in tqdm(range(iterations)):
        results = fitBandits.fitInfiniteSearch(baiAllocations.UCB.getBudget, f, d, maxBudget, batchSize,
                                               numEvalsPerGrad, minSamples, discountRate=discountRate,
                                               a=a, c=c, useTqdm=False, useSPSA=useSPSA)
        convergeDic = results[2]
        for s in convergeDic.keys():
            try:
                errors[s].append(convergeDic[s] - minimum)
            except:
                errors[s] = [convergeDic[s] - minimum]

    aveError = {}
    for s in errors.keys():
        aveError[s] = float(sum(errors[s])) / len(errors[s])

    aveErrorList.append(aveError)


def tempRestlessOCBA(aveErrorList, iterations, sharedParams, startPosList):
    f = sharedParams[0]
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
    discountFactor = sharedParams[12]
    windowLength = sharedParams[13]

    errors = {}
    for iteration in tqdm(range(iterations)):
        results = restlessBandits.restlessSearch(baiAllocations.discountedOCBA.getBudget, discountFactor, windowLength,
                                                 f, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                                 a=a, c=c, startPos=startPosList[iteration], useTqdm=False, useSPSA=useSPSA)

        convergeDic = results[2]
        for s in convergeDic.keys():
            try:
                errors[s].append(convergeDic[s] - minimum)
            except:
                errors[s] = [convergeDic[s] - minimum]

    aveError = {}
    for s in errors.keys():
        aveError[s] = float(sum(errors[s])) / len(errors[s])

    aveErrorList.append(aveError)


def tempRestlessInfiniteOCBA(aveErrorList, iterations, sharedParams, startPosList):
    f = sharedParams[0]
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
    discountFactor = sharedParams[12]
    windowLength = sharedParams[13]

    errors = {}
    for iteration in tqdm(range(iterations)):
        results = restlessBandits.restlessInfiniteSearch(baiAllocations.discountedOCBA.getBudget, discountFactor, windowLength,
                                                         f, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                                         a=a, c=c, useTqdm=False, useSPSA=useSPSA)
        convergeDic = results[2]
        for s in convergeDic.keys():
            try:
                errors[s].append(convergeDic[s] - minimum)
            except:
                errors[s] = [convergeDic[s] - minimum]

    aveError = {}
    for s in errors.keys():
        aveError[s] = float(sum(errors[s])) / len(errors[s])

    aveErrorList.append(aveError)


def tempRestlessUCB(aveErrorList, iterations, sharedParams, startPosList):
    f = sharedParams[0]
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
    discountFactor = sharedParams[12]
    windowLength = sharedParams[13]

    errors = {}
    for iteration in tqdm(range(iterations)):
        results = restlessBandits.restlessSearch(baiAllocations.discountedUCB.getBudget, discountFactor, windowLength,
                                                 f, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                          a=a, c=c, startPos=startPosList[iteration], useTqdm=False, useSPSA=useSPSA)
        convergeDic = results[2]
        for s in convergeDic.keys():
            try:
                errors[s].append(convergeDic[s] - minimum)
            except:
                errors[s] = [convergeDic[s] - minimum]

    aveError = {}
    for s in errors.keys():
        aveError[s] = float(sum(errors[s])) / len(errors[s])

    aveErrorList.append(aveError)


def tempRestlessInfiniteUCB(aveErrorList, iterations, sharedParams, startPosList):
    f = sharedParams[0]
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
    discountFactor = sharedParams[12]
    windowLength = sharedParams[13]

    errors = {}
    for iteration in tqdm(range(iterations)):
        results = restlessBandits.restlessInfiniteSearch(baiAllocations.discountedUCB.getBudget, discountFactor, windowLength,
                                                         f, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                                   a=a, c=c, useTqdm=False, useSPSA=useSPSA)
        convergeDic = results[2]
        for s in convergeDic.keys():
            try:
                errors[s].append(convergeDic[s] - minimum)
            except:
                errors[s] = [convergeDic[s] - minimum]

    aveError = {}
    for s in errors.keys():
        aveError[s] = float(sum(errors[s])) / len(errors[s])

    aveErrorList.append(aveError)


def tempTradOCBA(aveErrorList, iterations, sharedParams, startPosList):
    f = sharedParams[0]
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

    errors = {}
    for iteration in tqdm(range(iterations)):
        results = tradBandits.tradSearch(baiAllocations.OCBA.getBudget, f, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                          a=a, c=c, startPos=startPosList[iteration], useTqdm=False, useSPSA=useSPSA)

        convergeDic = results[2]
        for s in convergeDic.keys():
            try:
                errors[s].append(convergeDic[s] - minimum)
            except:
                errors[s] = [convergeDic[s] - minimum]

    aveError = {}
    for s in errors.keys():
        aveError[s] = float(sum(errors[s])) / len(errors[s])

    aveErrorList.append(aveError)


def tempTradInfiniteOCBA(aveErrorList, iterations, sharedParams, startPosList):
    f = sharedParams[0]
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

    errors = {}
    for iteration in tqdm(range(iterations)):
        results = tradBandits.tradInfiniteSearch(baiAllocations.OCBA.getBudget, f, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                                     a=a, c=c, useTqdm=False, useSPSA=useSPSA)
        convergeDic = results[2]
        for s in convergeDic.keys():
            try:
                errors[s].append(convergeDic[s] - minimum)
            except:
                errors[s] = [convergeDic[s] - minimum]

    aveError = {}
    for s in errors.keys():
        aveError[s] = float(sum(errors[s])) / len(errors[s])

    aveErrorList.append(aveError)


def tempTradUCB(aveErrorList, iterations, sharedParams, startPosList):
    f = sharedParams[0]
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

    errors = {}
    for iteration in tqdm(range(iterations)):
        results = tradBandits.tradSearch(baiAllocations.UCB.getBudget, f, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                          a=a, c=c, startPos=startPosList[iteration], useTqdm=False, useSPSA=useSPSA)
        convergeDic = results[2]
        for s in convergeDic.keys():
            try:
                errors[s].append(convergeDic[s] - minimum)
            except:
                errors[s] = [convergeDic[s] - minimum]

    aveError = {}
    for s in errors.keys():
        aveError[s] = float(sum(errors[s])) / len(errors[s])

    aveErrorList.append(aveError)


def tempTradInfiniteUCB(aveErrorList, iterations, sharedParams, startPosList):
    f = sharedParams[0]
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

    errors = {}
    for iteration in tqdm(range(iterations)):
        results = tradBandits.tradInfiniteSearch(baiAllocations.UCB.getBudget, f, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                                     a=a, c=c, useTqdm=False, useSPSA=useSPSA)
        convergeDic = results[2]
        for s in convergeDic.keys():
            try:
                errors[s].append(convergeDic[s] - minimum)
            except:
                errors[s] = [convergeDic[s] - minimum]

    aveError = {}
    for s in errors.keys():
        aveError[s] = float(sum(errors[s])) / len(errors[s])

    aveErrorList.append(aveError)


def tempUniform(aveErrorList, iterations, sharedParams, startPosList):
    f = sharedParams[0]
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

    errors = {}
    for iteration in tqdm(range(iterations)):
        results = uniformAlloc.uniformSearch(f, k, d, maxBudget, batchSize, numEvalsPerGrad,
                                             a=a, c=c, startPos=startPosList[iteration], useTqdm=False, useSPSA=useSPSA)

        convergeDic = results[2]
        for s in convergeDic.keys():
            try:
                errors[s].append(convergeDic[s] - minimum)
            except:
                errors[s] = [convergeDic[s] - minimum]

    aveError = {}
    for s in errors.keys():
        aveError[s] = float(sum(errors[s])) / len(errors[s])

    aveErrorList.append(aveError)

def tempMetaMax(aveErrorList, iterations, sharedParams, startPosList):
    f = sharedParams[0]
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

    errors = {}
    for iteration in tqdm(range(iterations)):
        results = metaMaxAlloc.metaMaxSearch(f, k, d, maxBudget, batchSize, numEvalsPerGrad,
                                             a=a, c=c, startPos=startPosList[iteration], useTqdm=False, useSPSA=useSPSA)

        convergeDic = results[2]
        for s in convergeDic.keys():
            try:
                errors[s].append(convergeDic[s] - minimum)
            except:
                errors[s] = [convergeDic[s] - minimum]

    aveError = {}
    for s in errors.keys():
        aveError[s] = float(sum(errors[s])) / len(errors[s])

    aveErrorList.append(aveError)

def tempMetaMaxInfinite(aveErrorList, iterations, sharedParams, startPosList):
    f = sharedParams[0]
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

    errors = {}
    for iteration in tqdm(range(iterations)):
        results = metaMaxAlloc.metaMaxInfiniteSearch(f, d, maxBudget, numEvalsPerGrad,
                                                     a=a, c=c, useTqdm=False, useSPSA=useSPSA)

        convergeDic = results[2]
        for s in convergeDic.keys():
            try:
                errors[s].append(convergeDic[s] - minimum)
            except:
                errors[s] = [convergeDic[s] - minimum]

    aveError = {}
    for s in errors.keys():
        aveError[s] = float(sum(errors[s])) / len(errors[s])

    aveErrorList.append(aveError)

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



def multiprocessSearch(numProcesses, iterations, func, sharedParams, processStartPos, endPath):
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
            # for i in range(numProcesses):
            #     print('Process p' + str(i+1) + ' is alive: {}'.format(processes[i].is_alive()))

            aveError = {}
            aveErrorListCopy = list(aveErrorList)
            for i in range(numProcesses):
                for sampleNum in aveErrorListCopy[i].keys():
                    try:
                        aveError[sampleNum] += aveErrorListCopy[i][sampleNum]
                    except:
                        aveError[sampleNum] = aveErrorListCopy[i][sampleNum]

            for sampleNum in aveError.keys():
                aveError[sampleNum] = aveError[sampleNum] / numProcesses

            with open(endPath, 'w') as fp:
                json.dump(aveError, fp)


def generateStartingPos(numProcesses, iterPerProcess, d, k, path, random=False):
    processStartPos = []
    print("Generating Starting Positions")
    for i in tqdm(range(numProcesses)):
        startPosList = []
        for j in range(iterPerProcess):
            if random:
                randoms = [gradientAllocation.randomParams(d) for i in range(k)]
                startPosList.append(randoms)

            else:
                startPosList.append(gradientAllocation.stratifiedSampling(d, k))

        processStartPos.append(startPosList)

    with open(path + "/startingPos.json", 'w') as jf:
        json.dump(processStartPos, jf)



def performMultiprocess(params, numProcesses, iterPerProcess, path, methods):
    fun = params[0]
    k = params[1]
    d = params[2]
    maxBudget = params[3]
    batchSize = params[4]
    numEvalsPerGrad = params[5]
    minSamples = params[6]

    minimum = params[7]
    discountRate = params[8]
    a = params[9]
    c = params[10]
    useSPSA = params[11]
    discountFactor = params[12]
    slidingWindow = params[13]


    with open(path + "/startingPos.json") as jf:
        processStartPos = json.load(jf)


    sharedParams = [fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                    minimum, discountRate, a, c, useSPSA, discountFactor, slidingWindow]

    if __name__ == '__main__':
        dir = path + "/"

        if methods[0]:
            print("FitOCBA")
            multiprocessSearch(numProcesses, iterPerProcess, tempFitOCBA, sharedParams, processStartPos,
                               dir + "FitOCBA.json")

        if methods[1]:
            print("FitInfiniteOCBA")
            multiprocessSearch(numProcesses, iterPerProcess, tempFitInfiniteOCBA, sharedParams, processStartPos,
                               dir + "FitInfiniteOCBA.json")

        if methods[2]:
            print("FitUCB")
            multiprocessSearch(numProcesses, iterPerProcess, tempFitUCB, sharedParams, processStartPos,
                               dir + "FitUCB.json")

        if methods[3]:
            print("FitInfiniteUCB")
            multiprocessSearch(numProcesses, iterPerProcess, tempFitInfiniteUCB, sharedParams, processStartPos,
                               dir + "FitInfiniteUCB.json")

        if methods[4]:
            print("RestlessOCBA")
            multiprocessSearch(numProcesses, iterPerProcess, tempRestlessOCBA, sharedParams, processStartPos,
                               dir + "RestlessOCBA.json")

        if methods[5]:
            print("RestlessInfiniteOCBA")
            multiprocessSearch(numProcesses, iterPerProcess, tempRestlessInfiniteOCBA, sharedParams, processStartPos,
                               dir + "RestlessInfiniteOCBA.json")

        if methods[6]:
            print("RestlessUCB")
            multiprocessSearch(numProcesses, iterPerProcess, tempRestlessUCB, sharedParams, processStartPos,
                               dir + "RestlessUCB.json")

        if methods[7]:
            print("RestlessInfiniteUCB")
            multiprocessSearch(numProcesses, iterPerProcess, tempRestlessInfiniteUCB, sharedParams, processStartPos,
                               dir + "RestlessInfiniteUCB.json")

        if methods[8]:
            print("TradOCBA")
            multiprocessSearch(numProcesses, iterPerProcess, tempTradOCBA, sharedParams, processStartPos,
                               dir + "TradOCBA.json")

        if methods[9]:
            print("TradInfiniteOCBA")
            multiprocessSearch(numProcesses, iterPerProcess, tempTradInfiniteOCBA, sharedParams, processStartPos,
                               dir + "TradInfiniteOCBA.json")

        if methods[10]:
            print("TradUCB")
            multiprocessSearch(numProcesses, iterPerProcess, tempTradUCB, sharedParams, processStartPos,
                               dir + "TradUCB.json")

        if methods[11]:
            print("TradInfiniteUCB")
            multiprocessSearch(numProcesses, iterPerProcess, tempTradInfiniteUCB, sharedParams, processStartPos,
                               dir + "TradInfiniteUCB.json")

        if methods[12]:
            print("Uniform")
            multiprocessSearch(numProcesses, iterPerProcess, tempUniform, sharedParams, processStartPos,
                               dir + "Uniform.json")

        if methods[13]:
            print("MetaMax")
            multiprocessSearch(numProcesses, iterPerProcess, tempMetaMax, sharedParams, processStartPos,
                               dir + "MetaMax.json")

        if methods[14]:
            print("MetaMaxInfinite")
            multiprocessSearch(numProcesses, iterPerProcess, tempMetaMaxInfinite, sharedParams, processStartPos,
                               dir + "MetaMaxInfinite.json")



def showMinimaHistory(dics, names, title):
    fig, ax = plt.subplots(1)

    for i in range(len(dics)):
        # jf = files[i]
        name = names[i]
        # aveError = json.load(jf)
        aveError = dics[i]
        try:

            x = list(aveError.keys())

        except:
            pass
        x = [int(i) for i in x]
        x.sort()
        y = [aveError[str(m)] for m in x]
        ax.plot(x, y, label=name)

    ax.title.set_text(title)
    ax.set_xlabel("Total Samples")
    ax.set_ylabel("Error")

    ax.legend(loc="upper right")
    # plt.semilogx()
    plt.show()


# paths = ['Results/efficientStrategiesComp/d2Random', 'Results/efficientStrategiesComp/d2Stratified',
#          'Results/efficientStrategiesComp/d10Random', 'Results/efficientStrategiesComp/d10Stratified']

# paths = ['Results/origComp/ackley/d2Random', 'Results/origComp/ackley/d2Stratified',
#          'Results/origComp/ackley/d10Random', 'Results/origComp/ackley/d10Stratified']

paths = ['Results/origComp/griewank/d2Random', 'Results/origComp/griewank/d2Stratified',
        'Results/origComp/griewank/d5Random', 'Results/origComp/griewank/d5Stratified',
        'Results/origComp/griewank/d10Random', 'Results/origComp/griewank/d10Stratified']

# paths = ['Results/origComp/griewank/d5Stratified',
#         'Results/origComp/griewank/d10Stratified']

# paths = ['Results/origComp/griewank/d10Random', 'Results/origComp/griewank/d10Stratified']


# paths = ['Results/efficientStrategiesComp/d10Random', 'Results/efficientStrategiesComp/d10Stratified']

# paths = ['Results/tests/test1', 'Results/tests/test2']

# path = "Results/efficientStrategiesComp/d2Random"
# path = "Results/tests/test2"

# if __name__ == '__main__':
#     orig_stdout = sys.stdout
#     orig_stderr = sys.stderr
#     f = open(path+'/consoleOutput.txt', 'w')
#     g = open(path+'/consoleOutput.txt', 'w')
#     sys.stdout = f
#     sys.stderr = g


"""
if __name__ == '__main__':
    for path in paths:
        print(f"Path is {path}")
        numProcesses, iterPerProcess, params, randomPos = paramPickler.readParams(path + "/params.txt")
        d = params[2]
        k = params[1]
        # generateStartingPos(numProcesses, iterPerProcess, d, k, path, random=randomPos)
        print()

        #         [fo,      foi,    fu,     fui,    ro,     roi,    ru,     rui,    to,     toi,    tu,     tui,    u,      mm,     mmi]
        # methods = [False ,  False,  False,  False,  True ,  True ,  True ,  True ,  True ,  True ,  True ,  True ,  True ,  False , False]
        methods = [False ,  False,  False,  False,  False,  False,  False,  False,  False,  False,  False,  False,  False,  True, True]


        performMultiprocess(params, numProcesses, iterPerProcess, path, methods)

        for i in range(5):
            print()
# """

# """
# path =  'Results/tests/test1'
for path in paths:
    allFileNames = os.listdir(path)
    # fileNames = ["metaMax.json", "tradOCBA.json", "tradUCB.json",
    #              "uniform.json", "metaMaxInfinite.json"]
    fileNames = [fileName for fileName in allFileNames if fileName.endswith(".json")
                 and fileName != "startingPos.json"]

    names = [fileName[:-5] for fileName in fileNames]
    dics = []
    for fileName in fileNames:
        with open(path + "\\" + fileName) as jf:
            dics.append(json.load(jf))
    print(dics)
    title = path
    showMinimaHistory(dics, names, title)
# """

