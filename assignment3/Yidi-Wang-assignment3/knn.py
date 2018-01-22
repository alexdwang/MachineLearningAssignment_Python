import scipy.io as sio
import scipy as sp
import numpy as np
import sys
import random
import operator
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
    # print (data.keys())
    return data["test_data"], data["test_label"], data["train_data"], data["train_label"]

def kford_split(dataset, kfold):
    trainsetindex = [[] for i in range(kfold)]
    copy = [i for i in range(len(dataset))]
    while len(copy) > 0:
        for i in range(kfold):
            if len(copy) == 0:
                break
            index = random.randrange(len(copy))
            trainsetindex[i].append(copy.pop(index))
    return trainsetindex

def calculatDis(val, train):
    test = val
    size = len(train)
    diffbetween = (np.tile(test, (size, 1)) - train) / 1.0
    diffsquaremat = diffbetween**2
    # print(type(diffsquaremat[0][0]))
    distances = diffsquaremat.sum(axis=1)
    sorteddistances = distances.argsort()
    return sorteddistances

def classify(sorteddistances, labels, k):
    count = {}
    for i in range(k):
        index = sorteddistances[i]
        label = labels[index][0]
        count[label] = count.get(label, 0) + 1
    sortedcount = sorted(count.items(), key = operator.itemgetter(1), reverse = True)
    return sortedcount[0][0]

def pca_fit(dataMat, n):
    meanVal = np.mean(dataMat, axis=0)
    newData = dataMat - meanVal
    covMat = np.cov(newData, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValIndice = np.argsort(eigVals)
    n_eigValIndice = eigValIndice[-1:-(n + 1):-1]
    n_eigVect = eigVects[:, n_eigValIndice]
    lowDataMat = newData * n_eigVect
    return lowDataMat, n_eigVect

def pca_transform(dataMat, n_eigVect):
    meanVal = np.mean(dataMat, axis=0)
    newData = dataMat - meanVal
    lowDataMat = newData * n_eigVect
    return lowDataMat

def plotfigure(k, noPCA, withPCA):
    plt.figure(1, figsize=(8,4))
    plt.plot(k, noPCA, 'b*')
    plt.plot(k, noPCA, 'r',label="KNN without PCA")
    plt.xlabel("K value")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper right')

    plt.figure(2, figsize=(8, 4))
    plt.plot(k, withPCA, 'b*')
    plt.plot(k, withPCA, 'r', label="KNN with PCA")
    plt.xlabel("K value")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper right')

    plt.figure(3, figsize=(8, 4))
    plt.plot(k, noPCA, 'b*')
    plt.plot(k, noPCA, 'r', label="KNN without PCA")
    plt.plot(k, withPCA, 'b*')
    plt.plot(k, withPCA, 'g', label="KNN with PCA")
    plt.xlabel("K value")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper right')

    plt.show()
    return

def main():
    testdata, testlabel, traindata, trainlabel = loadMatrix("knn_data.mat")

    ks = [2 * i + 1 for i in range(9)]
    kfold = 5
    trainsetindex = kford_split(traindata, kfold)

    #   knn without PCA
    correct = {}.fromkeys(ks, 0)
    cur = 0
    for i in range(kfold):
        copy = list(trainsetindex)
        val = list(copy.pop(i))
        tra = [x for j in copy for x in j]
        train = [traindata[i] for i in tra]
        trainlabels = [trainlabel[i] for i in tra]
        for aval in val:
            cur += 1
            sorteddistances = calculatDis(traindata[aval], train)
            for k in ks:
                valresult = classify(sorteddistances , trainlabels, k)
                # print("index=",aval,"  k=",k,"  predict=",valresult,"  true=", trainlabel[aval][0], "currentacc=", correct[k]/cur) if cur%1000==0 else 0
                if valresult == trainlabel[aval][0]:
                    correct[k] += 1
    choose = 0
    curacc = 0
    print("KNN without PCA")
    accuracies = [0 for k in ks]
    for k in ks:
        accuracy = correct[k] / len(traindata)
        accuracies[int((k - 1) / 2)] = accuracy
        if curacc < accuracy:
            choose = k
            curacc = accuracy
        print ("k = ",k,": ",accuracy)
    print("Choose k = ",choose, "whose accuracy is: ",curacc)
    correctness = 0
    i = 0
    for atest in testdata:
        sorteddistances = calculatDis(atest, traindata)
        testresult = classify(sorteddistances, trainlabel, k = choose)
        if testresult == testlabel[i][0]:
            correctness += 1
        i += 1
    print("Test accuracy is: ",correctness / len(testdata))

#   knn with PCA
    traindata, n_eigVect = pca_fit(traindata, n=50)
    testdata = pca_transform(testdata,n_eigVect)
    traindata = np.asarray(traindata)
    testdata = np.asarray(testdata)

    correct = {}.fromkeys(ks, 0)
    for i in range(kfold):
        copy = list(trainsetindex)
        val = list(copy.pop(i))
        tra = [x for j in copy for x in j]
        train = [traindata[i] for i in tra]
        trainlabels = [trainlabel[i] for i in tra]
        for aval in val:
            sorteddistances = calculatDis(traindata[aval], train)
            for k in ks:
                valresult = classify(sorteddistances, trainlabels, k)
                if valresult == trainlabel[aval][0]:
                    correct[k] += 1
    choose = 0
    curacc = 0
    print("KNN with PCA")
    accuracies2 = [0 for k in ks]
    for k in ks:
        accuracy = correct[k] / len(traindata)
        accuracies2[int((k - 1) / 2)] = accuracy
        if curacc < accuracy:
            choose = k
            curacc = accuracy
        print ("k = ", k, ": ", accuracy)
    print("Choose k = ", choose, "whose accuracy is: ", curacc)
    correctness = 0
    i = 0
    for atest in testdata:
        sorteddistances = calculatDis(atest, traindata)
        testresult = classify(sorteddistances, trainlabel, k=choose)
        if testresult == testlabel[i][0]:
            correctness += 1
        i += 1
    print("Test accuracy is: ", correctness / len(testdata))

    # print (len(val), " ", len(tra))
    plotfigure(ks, accuracies, accuracies2)
    return

if __name__ == "__main__":
    sys.exit(main())
