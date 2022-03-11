import json
import os
import statistics

import numpy as np
import matplotlib.pyplot as plt

def normalizeDic(dic):

    minimum = min(list(dic.values()))
    maximum = max(list(dic.values()))
    scale = maximum-minimum

    normalized = {}
    for key in dic.keys():
        normalized[key] = (dic[key]-minimum)/scale

    return normalized

resultsDir = 'Results/origComp'
# resultsDir = 'Results/origComp2'
paths = ['Ackley', 'Griewank', 'Rastrigin']
# paths = ['AckleyRandom', 'GriewankRandom', 'RastriginRandom']

# paths = ['griewank/2dim', 'griewank/5dim', 'griewank/10dim', 'griewank/20dim']
paths = [resultsDir + "/" + path for path in paths]

figDic = {}
for i in range(len(paths)):
    figDic[paths[i]] = i


dimensions = [2,5,10,20]
funcScores = {}

dimScores = {}

# produces the "dimScores" dictionary that has all of the scores (0 to 3) for every dimension, strategy
for dim in dimensions:
    stratScores = {'EEUniform': 0, 'MetaMax': 0, 'MetaMaxInfinite': 0, 'RestlessOCBA': 0, 'RestlessOCBAInfinite': 0,
                   'RestlessUCB': 0, 'RestlessUCBInfinite': 0, 'TradOCBA': 0, 'TradOCBAInfinite': 0, 'TradUCB': 0,
                   'TradUCBInfinite': 0, 'Uniform': 0, 'UniformInfinite': 0}
    # print(f"{dim} Dimensions:")
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

        normScores = normalizeDic(scores)

        for key in scores.keys():
            stratScores[key] += normScores[key]

    dimScores[dim] = stratScores
    # print(stratScores)
    # funcScores[path[17:]] = scores

    # print(funcScores)

print(dimScores)

# data = [list(dimScores[2].values()), list(dimScores[5].values()), list(dimScores[10].values()), list(dimScores[20].values())]
data = {2:[3-val for val in list(dimScores[2].values())], 5:[3-val for val in list(dimScores[5].values())], 10:[3-val for val in list(dimScores[10].values())],
        20:[3-val for val in list(dimScores[20].values())]}
print(data)


# source: https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars
def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, [str(key) + " Dimensions" for key in data.keys()])

    ax.set_xticks(range(13))
    ax.set_xticklabels((
                       'EEUniform', 'MetaMax', 'MetaMaxInfinite', 'RestlessOCBA', 'RestlessOCBAInfinite', 'RestlessUCB',
                       'RestlessUCBInfinite', 'TradOCBA', 'TradOCBAInfinite', 'TradUCB', 'TradUCBInfinite', 'Uniform',
                       'UniformInfinite'))
    for tick in ax.get_xticklabels():
        tick.set_rotation(70)
if __name__ == "__main__":

    # Usage example:
    # data = {
    #     "a": [1, 2, 3, 2, 1],
    #     "b": [2, 3, 4, 3, 1],
    #     "c": [3, 2, 1, 4, 2],
    #     "d": [5, 9, 2, 1, 8],
    #     "e": [1, 3, 2, 2, 3],
    #     "f": [4, 3, 1, 1, 4],
    # }

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.3)
    bar_plot(ax, data, total_width=.7, single_width=.9)
    plt.show()

"""
# colors=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'm', 'limegreen', 'bisque', 'lime', 'lightcoral', 'gold']
#
#
# X = np.arange(4)
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
#
# ax.bar(X + 0.00, data[0], color = 'b', width = 0.2)
# ax.bar(X + 0.2, data[1], color = 'g', width = 0.2)
# ax.bar(X + 0.4, data[2], color = 'r', width = 0.2)
# ax.bar(X + 0.6, data[3], color = 'o', width = 0.2)
#

N = 4
ind = np.arange(N)  # the x locations for the groups
width = 0.06       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

vals1 = data[0]
rects1 = ax.bar(ind, vals1, width, color='r')
vals2 = data[1]
rects2 = ax.bar(ind+width, vals2, width, color='g')
vals3 = data[2]
rects3 = ax.bar(ind, vals3, width, color='b')
vals4 = data[3]
rects4 = ax.bar(ind, vals4, width, color='o')

ax.set_ylabel('Scores')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('EEUniform', 'MetaMax', 'MetaMaxInfinite', 'RestlessOCBA', 'RestlessOCBAInfinite', 'RestlessUCB', 'RestlessUCBInfinite', 'TradOCBA', 'TradOCBAInfinite', 'TradUCB', 'TradUCBInfinite', 'Uniform', 'UniformInfinite') )
ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('y', 'z', 'k') )

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

"""



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


#
# print(orderDic(dimScores[2]))
# print(orderDic(dimScores[5]))
# print(orderDic(dimScores[10]))
# print(orderDic(dimScores[20]))
