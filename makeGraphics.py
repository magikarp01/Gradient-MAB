import statistics

import functions
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import multiprocessing as mp
import os
from results import showMinimaHistory

# paths = ['Results/origComp2/ackley2/d2Random', 'Results/origComp2/ackley2/d5Random',
#          'Results/origComp2/ackley2/d10Random', 'Results/origComp2/griewank2/d2Random',
#          'Results/origComp2/griewank2/d5Random', 'Results/origComp2/griewank2/d10Random',
#          'Results/origComp2/rastrigin/d2Random', 'Results/origComp2/rastrigin/d5Random',
#          'Results/origComp2/rastrigin/d10Random']

# funcDics is 2d array, 6 arrays of the results
# names is metaMax, metaMaxInfinite, etc
# titles has 6 titles for each

# funcDics is 2d array, 6 arrays of the results
# names is metaMax, metaMaxInfinite, etc
# titles has 6 titles for each


def displayResults(funcDics, figNum, names, wholeTitle,
                   titles=['d2Random', 'd5Random', 'd10Random', ],
                   colorMap = {'MetaMax': 'blue', 'MetaMaxInfinite': 'orange', 'RestlessInfiniteOCBA': 'green', 'RestlessInfiniteUCB': 'red', 'RestlessOCBA': 'purple', 'RestlessUCB': 'brown', 'TradInfiniteOCBA': 'pink', 'TradInfiniteUCB': 'gray', 'TradOCBA': 'olive', 'TradUCB': 'cyan', 'Uniform': 'm'}):
    plt.figure(figNum)
    plt.suptitle(wholeTitle)

    for i in range(3):
        plt.subplot(3, 2, 2*i+1)
        plt.title(titles[i])
        plt.xlabel("Total Samples")
        plt.ylabel("Error")

        # resultDic = dictionary mapping names to dictionary
        resultDic = funcDics[titles[i]]

        for j in range(len(names)):
            name = names[j]
            aveError = resultDic[name]
            x = list(aveError.keys())

            x = [int(val) for val in x]
            x.sort()
            y = [aveError[str(m)] for m in x]
            plt.plot(x, y, label=name, color = colorMap[name])



        plt.subplot(3, 2, 2*i+2)
        plt.title("Zoomed In")
        plt.xlabel("Total Samples")
        plt.ylabel("Error")

        # resultDic = dictionary mapping names to dictionary
        resultDic = funcDics[titles[i]]

        for j in range(len(names)):
            name = names[j]
            aveError = resultDic[name]
            x = list(aveError.keys())

            x = [int(val) for val in x]
            x.sort()
            y = [aveError[str(m)] for m in x]
            plt.plot(x, y, label=name, color = colorMap[name], linewidth=.5)


    # plt.legend(loc='upper right', bbox_to_anchor=(0, -0.1, .97, 1),
    #            bbox_transform=plt.gcf().transFigure)

    # plt.subplots_adjust(left=.07, right=.82, hspace=.5, wspace=.15)

    plt.legend(loc='upper right', bbox_to_anchor=(0, -0.1, .99, 1),
               bbox_transform=plt.gcf().transFigure)

    plt.subplots_adjust(left=.07, right=.8, hspace=.667, wspace=.225)



# figDic = {}
# for i in range(len(paths)):
#     figDic[paths[i]] = i

"""
figTitles = ["Ackley", "Griewank", "Rastrigin"]
paths = ['Results/origComp2/ackley2', 'Results/origComp2/griewank2', 'Results/origComp2/rastrigin2']
names = ['MetaMax', 'MetaMaxInfinite', 'RestlessInfiniteOCBA', 'RestlessInfiniteUCB', 'RestlessOCBA', 'RestlessUCB',
         'TradInfiniteOCBA', 'TradInfiniteUCB', 'TradOCBA', 'TradUCB', 'Uniform']

for i in range(len(paths)):
    print(paths[i])

    path = paths[i]
    # path = 'Results/origComp2/griewank2'
    direcs = os.listdir(path)
    funcDics = {}
    for folder in direcs:
        print(folder)
        direcPath = path + '/' + folder

        allFileNames = os.listdir(direcPath)
        fileNames = [fileName for fileName in allFileNames if fileName.endswith(".json")
                 and fileName != "startingPos.json" and fileName[:-5] in names]

        allocRankings = {}

        resultDic = {}
        for fileName in fileNames:
            with open(direcPath + "/" + fileName) as jf:
                loadedDic = json.load(jf)
                resultDic[fileName[:-5]] = loadedDic

                allocRankings[fileName[:-5]] = loadedDic[sorted(list(loadedDic.keys()))[-1]]
        print(allocRankings)
        print()
        print()

        funcDics[folder] = resultDic

    displayResults(funcDics, 1, names, figTitles[i] + " Function")
    plt.show()
# """


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


ackleyResultsDic = {
    2: {'MetaMax': 5.142130516465792, 'MetaMaxInfinite': 1.2275161808463364, 'RestlessInfiniteOCBA': 1.0285956605665214, 'RestlessInfiniteUCB': 1.0305608509611077, 'RestlessOCBA': 4.856316686710289, 'RestlessUCB': 4.748410779957973, 'TradInfiniteOCBA': 1.243187920725253, 'TradInfiniteUCB': 2.6968480616061035, 'TradOCBA': 4.653468973489789, 'TradUCB': 7.048226186938908, 'Uniform': 5.1323837418591465},
    5: {'MetaMax': 11.845276107898236, 'MetaMaxInfinite': 8.418489263104465, 'RestlessInfiniteOCBA': 7.0836240010509535, 'RestlessInfiniteUCB': 7.168895250728522, 'RestlessOCBA': 10.262813734219582, 'RestlessUCB': 10.160642665732173, 'TradInfiniteOCBA': 3.8643712514156743, 'TradInfiniteUCB': 4.617879978917879, 'TradOCBA': 9.870027354969096, 'TradUCB': 11.45864159457829, 'Uniform': 12.8008737799399},
    10:{'MetaMax': 17.186727920116066, 'MetaMaxInfinite': 16.667184148271634, 'RestlessInfiniteOCBA': 13.71321773563275, 'RestlessInfiniteUCB': 14.018124762852732, 'RestlessOCBA': 13.795417297626786, 'RestlessUCB': 13.584382168681426, 'TradInfiniteOCBA': 12.652432012106269, 'TradInfiniteUCB': 15.232946938112743, 'TradOCBA': 12.215457275921306, 'TradUCB': 16.68918521079151, 'Uniform': 18.221140982051317}
}

griewankResultsDic = {
    2: {'MetaMax': 0.11265629704067212, 'MetaMaxInfinite': 0.0008176712368932293, 'RestlessInfiniteOCBA': 0.0069632530202138575, 'RestlessInfiniteUCB': 0.004042336335247475, 'RestlessOCBA': 0.10534977635748705, 'RestlessUCB': 0.10557853970633224, 'TradInfiniteOCBA': 0.027073404608893185, 'TradInfiniteUCB': 0.018083578569810823, 'TradOCBA': 0.15536452265239653, 'TradUCB': 0.22757814334164744, 'Uniform': 0.1061537170877058},
    5: {'MetaMax': 0.3495463692821085, 'MetaMaxInfinite': 0.3322199342027433, 'RestlessInfiniteOCBA': 0.2512731703818248, 'RestlessInfiniteUCB': 0.2587744944269732, 'RestlessOCBA': 0.3145189963455965, 'RestlessUCB': 0.31103583567208287, 'TradInfiniteOCBA': 0.2698040891420947, 'TradInfiniteUCB': 0.3686817419149444, 'TradOCBA': 0.3847381241118367, 'TradUCB': 0.3727318850705964, 'Uniform': 0.37709907368233875},
    10:{'MetaMax': 0.1781792683269287, 'MetaMaxInfinite': 0.034558292599225576, 'RestlessInfiniteOCBA': 0.011384195699088034, 'RestlessInfiniteUCB': 0.002754137239006046, 'RestlessOCBA': 0.04079437127385921, 'RestlessUCB': 0.004453132319881343, 'TradInfiniteOCBA': 0.021312701085201488, 'TradInfiniteUCB': 0.12760880190260257, 'TradOCBA': 0.036164839285480296, 'TradUCB': 0.1559254277222373, 'Uniform': 0.6270412548662048}
}

rastriginResultsDic = {
    2: {'MetaMax': 1.9618278728169807, 'MetaMaxInfinite': 1.8583160574099102, 'RestlessInfiniteOCBA': 2.009714892857801, 'RestlessInfiniteUCB': 1.996362781801071, 'RestlessOCBA': 2.039605186201277, 'RestlessUCB': 1.9406019333119917, 'TradInfiniteOCBA': 1.842880861228877, 'TradInfiniteUCB': 1.854468773913854, 'TradOCBA': 2.0859128174602644, 'TradUCB': 3.125092924097806, 'Uniform': 2.1277304125759264},
    5: {'MetaMax': 25.165631695964894, 'MetaMaxInfinite': 21.688549719561287, 'RestlessInfiniteOCBA': 22.009779767931548, 'RestlessInfiniteUCB': 22.144194878322764, 'RestlessOCBA': 24.848613744608528, 'RestlessUCB': 24.531877572895347, 'TradInfiniteOCBA': 20.946763102327743, 'TradInfiniteUCB': 22.241352627716655, 'TradOCBA': 23.365350042229807, 'TradUCB': 19.479190320076647, 'Uniform': 25.598856989505553},
    10:{'MetaMax': 64.99845807273115, 'MetaMaxInfinite': 65.33731152362442, 'RestlessInfiniteOCBA': 70.92547359160803, 'RestlessInfiniteUCB': 70.92547359160803, 'RestlessOCBA': 65.01395158555916, 'RestlessUCB': 65.01395158555916, 'TradInfiniteOCBA': 71.41282750269548, 'TradInfiniteUCB': 71.17650340497545, 'TradOCBA': 65.01395158555916, 'TradUCB': 65.01395158555916, 'Uniform': 64.99927759576406}
}

def sumStandardDics(d):
    finDic = {}
    d1 = standardizeDic(ackleyResultsDic[d])
    d2 = standardizeDic(griewankResultsDic[d])
    d3 = standardizeDic(rastriginResultsDic[d])
    for key in d1.keys():
        finDic[key] = d1[key] + d2[key] + d3[key]
    return finDic

print(sumStandardDics(2))
orderDic(sumStandardDics(2))
print()

print(sumStandardDics(5))
orderDic(sumStandardDics(5))
print()

print(sumStandardDics(10))
orderDic(sumStandardDics(10))
print()

"""
orderDic(
    {'MetaMax': 1.9618278728169807, 'MetaMaxInfinite': 1.8583160574099102, 'RestlessInfiniteOCBA': 2.009714892857801, 'RestlessInfiniteUCB': 1.996362781801071, 'RestlessOCBA': 2.039605186201277, 'RestlessUCB': 1.9406019333119917, 'TradInfiniteOCBA': 1.842880861228877, 'TradInfiniteUCB': 1.854468773913854, 'TradOCBA': 2.0859128174602644, 'TradUCB': 3.125092924097806, 'Uniform': 2.1277304125759264}
  )
print()

orderDic(
    {'MetaMax': 25.165631695964894, 'MetaMaxInfinite': 21.688549719561287, 'RestlessInfiniteOCBA': 22.009779767931548, 'RestlessInfiniteUCB': 22.144194878322764, 'RestlessOCBA': 24.848613744608528, 'RestlessUCB': 24.531877572895347, 'TradInfiniteOCBA': 20.946763102327743, 'TradInfiniteUCB': 22.241352627716655, 'TradOCBA': 23.365350042229807, 'TradUCB': 19.479190320076647, 'Uniform': 25.598856989505553}
  )
print()

orderDic(
    {'MetaMax': 64.99845807273115, 'MetaMaxInfinite': 65.33731152362442, 'RestlessInfiniteOCBA': 70.92547359160803, 'RestlessInfiniteUCB': 70.92547359160803, 'RestlessOCBA': 65.01395158555916, 'RestlessUCB': 65.01395158555916, 'TradInfiniteOCBA': 71.41282750269548, 'TradInfiniteUCB': 71.17650340497545, 'TradOCBA': 65.01395158555916, 'TradUCB': 65.01395158555916, 'Uniform': 64.99927759576406}
  )
print()

# """


