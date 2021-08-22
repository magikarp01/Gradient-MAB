# a file for the shared functions that all gradient allocation files use

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import gradDescent
import kriging


# range in [0, 1]
def randomParams(d):
    vec = [None] * d
    for i in range(d):
        vec[i] = random.random()
    return np.array(vec)

# each element has range in [0, 1]
# returns list of k vectors, d elements per vector
# splits up the d dimensional hypercube into k smaller hypercubes
# spaces out starting positions by placing them in the smaller hypercubes

def stratifiedSampling(d, k):
    numDivPerDim = math.ceil(float(k) ** (1.0 / d))
    totalDivs = numDivPerDim ** d
    randomVecs = []
    for i in range(totalDivs):
        randomVec = []
        for dim in range(d):
            pos = (i % (numDivPerDim ** (dim + 1))) // (numDivPerDim ** dim)
            pos += random.random()
            pos /= numDivPerDim
            randomVec.append(pos)
        randomVecs.append(randomVec)

    to_keep = set(random.sample(range(totalDivs), k))
    return [randomVecs[i] for i in to_keep]


# for displaying how the instances move on a graph
# colors is an array of length k,
def displayInstances1D(f, instances, ax, colors, yRange, fColor='b', lineWidth=3):
    ax.title.set_text("Instance Performance")

    ax.set_xlim(-.5, 1.5)
    ax.set_ylim(yRange[0], yRange[1])
    x = list(np.linspace(-.5, 1.5, 1000))
    y = [f([x1]) for x1 in x]
    ax.plot(x, y, color=fColor)
    for i in range(len(instances)):
        instance = instances[i]
        xArray = [i[0][0] for i in instance]
        yArray = [i[1] for i in instance]
        ax.plot(xArray, yArray, linewidth=lineWidth, color=colors[i], label="instance" + str(i))
        firstX = instance[0][0][0]
        firstY = instance[0][1]
        ax.plot(firstX, firstY, marker=".", markersize=15, color=colors[i])

    ax.legend(loc="upper right", bbox_to_anchor=(1.12, 1))


# for displaying how samples are allocated
def displaySamplingHistory(samplingDic, ax, colors):
    ax.title.set_text("Sampling History")
    ax.set_xlabel("Total Samples")
    ax.set_ylabel("Instance Samples")

    k = len(samplingDic[next(iter(samplingDic))])
    xArray = list(samplingDic.keys())
    ax.set_xlim(left=int(xArray[0]/1.1), right=int(xArray[-1]*1.1))

    for i in range(k):
        yArray = []
        for totalBudget in samplingDic.keys():
            yArray.append(samplingDic[totalBudget][i])
        ax.plot(xArray, yArray, color=colors[i], label="instance" + str(i))


# for displaying how the minima changes as the time goes on
def displayMinimaHistory(convergeDic, ax):
    ax.title.set_text("Minimum History")
    ax.set_xlabel("Total Samples")
    ax.set_ylabel("Minimum")

    x = convergeDic.keys()
    y = [convergeDic[m] for m in x]

    ax.plot(x, y)


# when fitting, we should make sure the function had a minimum
# or, if it is opposite, we should just have the value be unbounded


