# Results are averaged over 1008 runs
# Parameters are in the tempOCBA function
import time

import allocMethods
import functions
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import multiprocessing as mp

import gradientAllocation
import newBaiAllocations
import rewardModels
from meta import paramPickler
import generalBandits


fun = functions.ackley_adjusted


def multiprocessTesting(numProcesses, iterations, func, sharedParams, processStartPos, endPath):
    if __name__ == '__main__':
        start_times = [0]*numProcesses
        times = [0]*numProcesses
        with mp.Manager() as manager:
            aveErrorList = manager.list()
            # print("ID of main process: {}".format(os.getpid()))
            processes = []
            for i in range(numProcesses):
                startPosList = processStartPos[i]
                start_times[i] = time.time()
                processes.append(mp.Process(target=func, args=(aveErrorList, iterations, sharedParams, startPosList)))
                processes[i].start()

            for i in range(numProcesses):
                processes[i].join()
                times[i] = time.time()


            print("All processes finished execution!")
            print(times)

            # check if processes are alive
            # for i in range(numProcesses):
            #     print('Process p' + str(i+1) + ' is alive: {}'.format(processes[i].is_alive()))


            # linear interpolation:
            for m in range(len(aveErrorList)):
                aveErrorDic = aveErrorList[m]
                aveErrorList[m] = gradientAllocation.linearInterp(aveErrorDic, sharedParams[3])

    # averaging
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


# models = [rewardModels.fit, rewardModels.restless, rewardModels.trad]
models = [rewardModels.restless, rewardModels.trad]
modelNames = ["Restless", "Trad"]
mabPolicies = [newBaiAllocations.OCBA.getBudget, newBaiAllocations.UCB.getBudget]
policyNames = ["OCBA", "UCB"]

methods = []
methodNames = []

for modelNum in range(len(models)):
    for policyNum in range(len(mabPolicies)):
        allocMethod = allocMethods.baiAllocate(models[modelNum], mabPolicies[policyNum])
        methods.append(allocMethod)
        methodNames.append(modelNames[modelNum] + policyNames[policyNum])

methods.append(allocMethods.uniform)
methods.append(allocMethods.metaMax)


def performMultiprocess(params, numProcesses, iterPerProcess, path, whichMethods):
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

        for i in range(len(whichMethods)):
            if whichMethods[i]:
                print(methodNames[i])
                # make a function that takes in a method and performs the search
                multiprocessTesting(numProcesses, iterPerProcess, ___, sharedParams,
                                    processStartPos, dir + methodNames[i] + ".json")


def showMinimaHistory(dics, names, title, figNum, colors=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'm', 'limegreen', 'bisque', 'lime', 'lightcoral', 'gold']):
    plt.figure(figNum)

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
        plt.plot(x, y, label=name, color = colors[i])

    plt.title(title)
    plt.xlabel("Total Samples")
    plt.ylabel("Error")

    plt.legend(loc="upper right")
    # plt.semilogx()
    # plt.show()


# paths = ['Results/efficientStrategiesComp/d2Random', 'Results/efficientStrategiesComp/d2Stratified',
#          'Results/efficientStrategiesComp/d10Random', 'Results/efficientStrategiesComp/d10Stratified']

# paths = ['Results/origComp/ackley/d2Random', 'Results/origComp/ackley/d2Stratified',
#         'Results/origComp/ackley/d5Random', 'Results/origComp/ackley/d5Stratified',
#         'Results/origComp/ackley/d10Random', 'Results/origComp/ackley/d10Stratified',
#         'Results/origComp/griewank2/d2Random', 'Results/origComp/griewank2/d2Stratified',
#         'Results/origComp/griewank2/d5Random', 'Results/origComp/griewank2/d5Stratified',
#         'Results/origComp/griewank2/d10Random', 'Results/origComp/griewank2/d10Stratified',
#         'Results/origComp/rastrigin/d2Random', 'Results/origComp/rastrigin/d2Stratified',
#         'Results/origComp/rastrigin/d5Random', 'Results/origComp/rastrigin/d5Stratified',
#         'Results/origComp/rastrigin/d10Random', 'Results/origComp/rastrigin/d10Stratified']

# paths = ['Results/origComp2/ackley2/d2Random', 'Results/origComp2/ackley2/d5Random',
#          'Results/origComp2/ackley2/d10Random', 'Results/origComp2/griewank2/d2Random',
#          'Results/origComp2/griewank2/d5Random', 'Results/origComp2/griewank2/d10Random',
#          'Results/origComp2/rastrigin/d2Random', 'Results/origComp2/rastrigin/d5Random',
#          'Results/origComp2/rastrigin/d10Random']

# paths = ['Results/origComp2/ackley2/d2Random', 'Results/origComp2/ackley2/d5Random',
#          'Results/origComp2/ackley2/d10Random']

# paths = ['Results/origComp/griewank2/d2Random', 'Results/origComp/griewank2/d2Stratified',
#         'Results/origComp/griewank2/d5Random', 'Results/origComp/griewank2/d5Stratified',
#         'Results/origComp/griewank2/d10Random', 'Results/origComp/griewank2/d10Stratified']


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


# """
paths = ['Results/origComp2/ackley2/d20Random',
         'Results/origComp2/griewank2/d20Random',
         'Results/origComp2/rastrigin2/d20Random']

if __name__ == '__main__':
    for path in paths:
        print(f"Path is {path}")

        # params = [fun, k, d, maxBudget, batchSize, numEvalsPerGrad, minSamples, minimum, discountRate, a, c, useSPSA, discountFactor, slidingWindow]
        numProcesses, iterPerProcess, params, randomPos = paramPickler.readParams(path + "/params.txt")
        d = params[2]
        k = params[1]

        # numProcesses = 2
        # iterPerProcess = 2
        # params[9] = .001
        # batchSize = 1000
        # params[6] = 2*d+5
        # params[13] = 2*d+5
        generateStartingPos(numProcesses, iterPerProcess, d, k, path, random=randomPos)
        print()

        #         [fo,      foi,    fu,     fui,    ro,     roi,    ru,     rui,    to,     toi,    tu,     tui,    u,      mm,     mmi]
        # methods = [False ,  False,  False,  False,  True ,  False,  True ,  True ,  True ,  True ,  True ,  True ,  True ,  True , True]
        methods = [False ,  False,  False,  False,  True ,  True ,  True ,  True ,  True ,  True ,  True ,  True ,  True ,  True , True ]
        # methods = [True ,   False,  True ,  False,  False,  False,  False,  False,  False,  False,  False,  False,  False,  False,  False]
        # methods = [False ,  False,  False,  False,  True ,  True ,  False,  False,  True ,  True ,  False,  False,  False,  False,  False]

        performMultiprocess(params, numProcesses, iterPerProcess, path, methods)

        for i in range(5):
            print()
# """


"""

# paths = ['Results/origComp/ackley2/d2Random', 'Results/origComp/ackley2/d5Random',
#          'Results/origComp/ackley2/d10Random', 'Results/origComp/griewank2/d2Random',
#          'Results/origComp/griewank2/d5Random', 'Results/origComp/griewank2/d10Random',
#          'Results/origComp/rastrigin/d2Random', 'Results/origComp/rastrigin/d5Random',
#          'Results/origComp/rastrigin/d10Random']

paths = ['Results/origComp2/rastrigin/d2Random', 'Results/origComp2/rastrigin/d5Random',
         'Results/origComp2/rastrigin2/d10Random']

figDic = {}
for i in range(len(paths)):
    figDic[paths[i]] = i


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
    showMinimaHistory(dics, names, path, figDic[path])
plt.show()
# """

