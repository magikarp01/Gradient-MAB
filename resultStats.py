import json
import os
import statistics

resultsDir = 'Results/origComp'
paths = ['Ackley', 'Griewank', 'Rastrigin']
# paths = ['AckleyRandom', 'GriewankRandom', 'RastriginRandom']

# paths = ['griewank/2dim', 'griewank/5dim', 'griewank/10dim', 'griewank/20dim']
paths = [resultsDir + "/" + path for path in paths]

figDic = {}
for i in range(len(paths)):
    figDic[paths[i]] = i


dimensions = [2,5,10,20]
funcScores = {}
stratScores = {'EEUniform': 0, 'MetaMax': 0, 'MetaMaxInfinite': 0, 'RestlessOCBA': 0, 'RestlessOCBAInfinite': 0, 'RestlessUCB': 0, 'RestlessUCBInfinite': 0, 'TradOCBA': 0, 'TradOCBAInfinite': 0, 'TradUCB': 0, 'TradUCBInfinite': 0, 'Uniform': 0, 'UniformInfinite': 0}

for dim in dimensions:
    for path in paths:
        folder = path + "/" + str(dim) + "dim"

        allFileNames = os.listdir(folder)
        # fileNames = ["metaMax.json", "tradOCBA.json", "tradUCB.json",
        #              "uniform.json", "metaMaxInfinite.json"]
        fileNames = [fileName for fileName in allFileNames if fileName.endswith(".json")
                     and fileName != "startingPos.json"]

        names = [fileName[:-5] for fileName in fileNames]

        scores = {}

        for fileName in fileNames:
            with open(folder + "/" + fileName) as jf:
                dic = json.load(jf)
                scores[fileName[:-5]] = list(dic.items())[-1][1]

        funcScores[path[17:]] = scores

    print(funcScores)




def orderDic(dic):
    # inv_dic = {v: k for k, v in dic.items()}
    # sortedKeys = sorted(list(inv_dic.keys()))
    # for i in sortedKeys:
    #     print(inv_dic[i])
    sortedDic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1])}
    for i in sortedDic.keys():
        print(i)
    print()
    for j in sortedDic.values():
        print(j)


def standardizeDic(dic):
    standardized = {}
    values = list(dic.values())
    mean = sum(values)/len(values)
    stdev = statistics.stdev(values)
    for key in dic.keys():
        standardized[key] = (dic[key]-mean)/stdev

    return standardized

def normalizeDic(dic):

    minimum = min(list(dic.values()))
    maximum = max(list(dic.values()))
    scale = maximum-minimum

    normalized = {}
    for key in dic.keys():
        normalized[key] = (dic[key]-minimum)/scale

    return normalized
#
# print(orderDic(dimScores[2]))
# print(orderDic(dimScores[5]))
# print(orderDic(dimScores[10]))
# print(orderDic(dimScores[20]))
