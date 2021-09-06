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
                    iterations, startPosList, minimum=0, a=.001, c=.001, useSPSA=False):
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


def getAveMetaMaxInfiniteError(fun, d, maxBudget, numEvalsPerGrad,
                    iterations, minimum=0, a=.001, c=.001, useSPSA=False):
    errors = {}
    for iteration in tqdm(range(iterations)):
        results = metaMaxAlloc.metaMaxInfiniteSearch(fun, d, maxBudget, numEvalsPerGrad,
                        a=a, c=c, useSPSA=useSPSA)
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

def tempFitOCBA(aveErrorList, iterations, sharedParams, startPosList):
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

    errors, aveError = getAveFitOCBAError(fun, k, d, maxBudget, batchSize,
                                          numEvalsPerGrad, minSamples, iterations, startPosList,
                                          minimum=minimum, discountRate=discountRate, a=a, c=c, useSPSA=useSPSA)
    aveErrorList.append(aveError)


def tempTradOCBA(aveErrorList, iterations, sharedParams, startPosList):
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
    errors, aveError = getAveTradOCBAError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                           iterations, startPosList,
                                          minimum=minimum, a=a, c=c, useSPSA=useSPSA)
    aveErrorList.append(aveError)


def tempFitUCB(aveErrorList, iterations, sharedParams, startPosList):
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
    errors, aveError = getAveFitUCBError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                         iterations, startPosList,
                                         minimum=minimum, discountRate=discountRate, a=a, c=c, useSPSA=useSPSA)
    aveErrorList.append(aveError)


def tempTradUCB(aveErrorList, iterations, sharedParams, startPosList):
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
    errors, aveError = getAveTradUCBError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                                          iterations, startPosList,
                                         minimum=minimum, a=a, c=c, useSPSA=useSPSA)
    aveErrorList.append(aveError)


def tempUniform(aveErrorList, iterations, sharedParams, startPosList):
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
    errors, aveError = getAveUniformError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad,
                                          iterations, startPosList,
                                       minimum=minimum, a=a, c=c, useSPSA=useSPSA)

    aveErrorList.append(aveError)


def tempMetaMax(aveErrorList, iterations, sharedParams, startPosList):
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
    errors, aveError = getAveMetaMaxError(fun, k, d, maxBudget, batchSize, numEvalsPerGrad,
                                          iterations, startPosList,
                                       minimum=minimum, a=a, c=c, useSPSA=useSPSA)

    aveErrorList.append(aveError)


def tempMetaMaxInfinite(aveErrorList, iterations, sharedParams, startPosList):
    fun = sharedParams[0]
    d = sharedParams[2]
    maxBudget = sharedParams[3]
    numEvalsPerGrad = sharedParams[5]

    minimum = sharedParams[7]

    a = sharedParams[9]
    c = sharedParams[10]
    useSPSA = sharedParams[11]
    errors, aveError = getAveMetaMaxInfiniteError(fun, d, maxBudget, numEvalsPerGrad,
                    iterations, minimum=minimum, a=a, c=c, useSPSA=useSPSA)

    aveErrorList.append(aveError)


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
            for i in range(numProcesses):
                print('Process p' + str(i+1) + ' is alive: {}'.format(processes[i].is_alive()))

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


def performMultiprocess(numProcesses, iterPerProcess, path):
    fun = functions.griewank_adjusted
    k = 5
    d = 10
    maxBudget = 10000
    batchSize = 50
    numEvalsPerGrad = 2
    minSamples = 3

    minimum = -1
    discountRate = .8
    a = .002
    c = .000001
    useSPSA = True


    # processStartPos = []
    # print("Generating Starting Positions")
    # for i in tqdm(range(numProcesses)):
    #     startPosList = []
    #     for j in range(iterPerProcess):
    #         # randoms = [gradientAllocation.randomParams(d) for i in range(k)]
    #         # startPosList.append(randoms)
    #
    #         startPosList.append(gradientAllocation.stratifiedSampling(d, k))
    #     processStartPos.append(startPosList)
    #
    # with open(path + "/startingPos.json", 'w') as jf:
    #     json.dump(processStartPos, jf)


    with open(path + "/startingPos.json") as jf:
        processStartPos = json.load(jf)


    sharedParams = [fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples,
                    minimum, discountRate, a, c, useSPSA]

    if __name__ == '__main__':
        dir = path + "/"

        # # print("Fit OCBA")
        # # multiprocessSearch(numProcesses, iterPerProcess, tempFitOCBA, sharedParams, processStartPos, dir+"fitOCBA.json")
        #
        # # print("Fit UCB")
        # # multiprocessSearch(numProcesses, iterPerProcess, tempFitUCB, sharedParams, processStartPos, dir + "fitUCB.json")
        #
        print("Trad OCBA")
        multiprocessSearch(numProcesses, iterPerProcess, tempTradOCBA, sharedParams, processStartPos, dir + "tradOCBA.json")

        print("Trad UCB")
        multiprocessSearch(numProcesses, iterPerProcess, tempTradUCB, sharedParams, processStartPos, dir + "tradUCB.json")
        #
        print("MetaMax")
        multiprocessSearch(numProcesses, iterPerProcess, tempMetaMax, sharedParams, processStartPos, dir + "metaMax.json")
        #

        print("MetaMaxInfinite")
        multiprocessSearch(numProcesses, iterPerProcess, tempMetaMaxInfinite, sharedParams, processStartPos, dir + "metaMaxInfinite.json")

        print("Uniform")
        multiprocessSearch(numProcesses, iterPerProcess, tempUniform, sharedParams, processStartPos, dir + "uniform.json")


def showMinimaHistory(dics, names):
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

    ax.title.set_text("Average Error History")
    ax.set_xlabel("Total Samples")
    ax.set_ylabel("Error")

    ax.legend(loc="upper right")
    # plt.semilogx()
    plt.show()



path = "Results/averageErrors/d10GriewankSPSA"
performMultiprocess(15, 667, path)


# allFileNames = os.listdir(path)
# # fileNames = ["metaMax.json", "tradOCBA.json", "tradUCB.json",
# #              "uniform.json", "metaMaxInfinite.json"]
# fileNames = [fileName for fileName in allFileNames if fileName.endswith(".json")
#              and fileName != "startingPos.json"]
#
# names = [fileName[:-5] for fileName in fileNames]
# dics = []
# for fileName in fileNames:
#     with open(path + "\\" + fileName) as jf:
#         dics.append(json.load(jf))
# # print(dics)
# showMinimaHistory(dics, names)

