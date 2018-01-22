import scipy.io as sio
import scipy as sp
import numpy as np
import sys
import random
import matplotlib.pyplot as plt

'''
' Author: Yidi Wang
' language version: Python 3.6.3
' All the packages above (excluding default packages) are downloaded from Laboratory for Fluorescence Dynamics, University of California, Irvine.
' link: http://www.lfd.uci.edu/~gohlke/pythonlibs
' Please update your packages to cp36-win_amd64 version if you can not run this code
'''

def loadMatrix(filename):
    data =  sio.loadmat(filename)
    np.set_printoptions(threshold=np.inf)
    data.pop("__header__")
    data.pop("__version__")
    data.pop("__globals__")
    # print (data["data"].shape)
    return data["data"]

def kmeans(data, k):
    nSamples, m = data.shape
    startindice = random.sample([i for i in range(nSamples)], k)
    center = [data[i] for i in startindice]
    #print(center)
    cluster = np.zeros((nSamples, 2))
    Goon = True
    while Goon:
        Goon = False
        for i in range(nSamples):
            minDist = 100000.0
            minIndex = 0
            for j in range(k):
                distance = sum(np.square(data[i] - center[j]))
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            if cluster[i,0] != minIndex:
                Goon = True
            cluster[i,:] = minIndex, minDist
        if Goon:
            for j in range(k):
                points = []
                for i in range(nSamples):
                    if cluster[i, 0] == j:
                        points.append(data[i])
                center[j] = np.mean(points, axis = 0)
    OFV = 0
    for i in range(nSamples):
        OFV += cluster[i,1]
    return cluster, center, OFV

'''This method is used to initialize centers for kmeans++. I learned this method from stackoverflow'''
def centerpick(data,K):
    new_center = [data[0]]
    for k in range(1, K):
        D2 = sp.array([min([sp.inner(c - x, c - x) for c in new_center]) for x in data])
        probs = D2 / D2.sum()
        cumprobs = probs.cumsum()
        r = sp.rand()
        for j,p in enumerate(cumprobs):
            if r < p:
                i = j
                break
        new_center.append(data[i])
    # pick the most distant point as center at a time   --- this is not the optimal solution, but it is on lecture note
    # nSamples, m = data.shape
    # c = len(center)
    # LargestDist = 0
    # new_center = data[0]
    # for i in range(nSamples):
    #     minDist = 100000.0
    #     for j in range(c):
    #         distance = sum(np.square(data[i] - center[j]))
    #         if distance < minDist:
    #             minDist = distance
    #     if LargestDist < minDist:
    #         new_center = data[i]
    return new_center

def kmeans_pp(data, k):
    nSamples, m = data.shape
    center = centerpick(data,k)
    cluster = np.zeros((nSamples, 2))
    Goon = True
    while Goon:
        Goon = False
        for i in range(nSamples):
            minDist = 100000.0
            minIndex = 0
            for j in range(k):
                distance = sum(np.square(data[i] - center[j]))
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            if cluster[i,0] != minIndex:
                Goon = True
            cluster[i,:] = minIndex, minDist
        if Goon:
            for j in range(k):
                points = []
                for i in range(nSamples):
                    if cluster[i, 0] == j:
                        points.append(data[i])
                if points:
                    center[j] = np.mean(points, axis = 0)
    OFV = 0
    for i in range(nSamples):
        OFV += cluster[i,1]
    return cluster, center, OFV

def plotfigure(k, OFV, OFVl_p):
    plt.figure(1, figsize=(8,4))
    plt.plot(k, OFV, 'b*',label="Kmeans")
    plt.plot(k, OFV, 'r')
    plt.xlabel("K value")
    plt.ylabel("Objective function value")
    plt.legend(loc='upper right')

    plt.figure(2, figsize=(8, 4))
    plt.plot(k, OFVl_p, 'b*', label="Kmeans++")
    plt.plot(k, OFVl_p, 'r')
    plt.xlabel("K value")
    plt.ylabel("Objective function value")
    plt.legend(loc='upper right')

    plt.figure(3, figsize=(8, 4))
    plt.plot(k, OFV, 'b*')
    plt.plot(k, OFV, 'r', label="Kmeans")
    plt.plot(k, OFVl_p, 'b*')
    plt.plot(k, OFVl_p, 'g', label="Kmeans++")
    plt.xlabel("K value")
    plt.ylabel("Objective function value")
    plt.legend(loc='upper right')

    plt.show()
    return

def main():
    data = loadMatrix("kmeans_data.mat")    # 7195 * 21
    ks = [i for i in range(2,11)]
    OFVl = []
    OFVl_p = []
    # kmeans:
    for k in ks:
        cluster, center, OFV = kmeans(data, k)
        OFVl.append(OFV)
        print(OFV)
    # kmeans++:
    for k in ks:
        cluster, center, OFV = kmeans_pp(data, k)
        OFVl_p.append(OFV)
        print(OFV)
    plotfigure(ks, OFVl, OFVl_p)
    return

if __name__ == "__main__":
    sys.exit(main())
