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
            plt.plot(x, y, label=name, color = colorMap[name])


    plt.legend(loc='upper right', bbox_to_anchor=(0, -0.1, .97, 1),
               bbox_transform=plt.gcf().transFigure)

    plt.subplots_adjust(left=.07, right=.82, hspace=.5, wspace=.15)


# figDic = {}
# for i in range(len(paths)):
#     figDic[paths[i]] = i

# """
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


"""
def orderDic(dic):
    # inv_dic = {v: k for k, v in dic.items()}
    # sortedKeys = sorted(list(inv_dic.keys()))
    # for i in sortedKeys:
    #     print(inv_dic[i])
    sortedDic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1])}
    for i in sortedDic.keys():
        print(i)


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
