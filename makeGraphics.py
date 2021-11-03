import functions
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import multiprocessing as mp
import os

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
                   colors=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'm', 'limegreen']):
    plt.figure(figNum)
    plt.suptitle(wholeTitle)
    colorMap = {}
    for c in range(len(names)):
        colorMap[names[c]] = colors[c]

    for i in range(3):
        plt.subplot(3, 2, 2*i+1)
        plt.title(titles[i])
        plt.xlabel("Total Samples")
        plt.ylabel("Error")

        dics = funcDics[titles[i]]

        for j in range(len(names)):
            aveError = dics[j]
            name = names[j]
            x = list(aveError.keys())

            x = [int(val) for val in x]
            x.sort()
            y = [aveError[str(m)] for m in x]
            plt.plot(x, y, label=name, color = colorMap[name])



        plt.subplot(3, 2, 2*i+2)
        plt.title("Zoomed In")
        plt.xlabel("Total Samples")
        plt.ylabel("Error")

        dics = funcDics[titles[i]]

        for j in range(len(names)):
            aveError = dics[j]
            name = names[j]
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
for i in range(len(paths)):
    path = paths[i]
    # path = 'Results/origComp2/griewank2'
    direcs = os.listdir(path)
    funcDics = {}
    for folder in direcs:
        direcPath = path + '/' + folder

        allFileNames = os.listdir(direcPath)
        fileNames = [fileName for fileName in allFileNames if fileName.endswith(".json")
                 and fileName != "startingPos.json"]

        names = [fileName[:-5] for fileName in fileNames]
        dics = []
        for fileName in fileNames:
            with open(direcPath + "/" + fileName) as jf:
                dics.append(json.load(jf))

        print(names)
        funcDics[folder] = dics

    names = ['MetaMax', 'MetaMaxInfinite', 'RestlessInfiniteOCBA', 'RestlessInfiniteUCB', 'RestlessOCBA', 'RestlessUCB', 'TradInfiniteOCBA', 'TradInfiniteUCB', 'TradOCBA', 'TradUCB', 'Uniform']
    displayResults(funcDics, 1, names, figTitles[i] + " Function")
    plt.show()