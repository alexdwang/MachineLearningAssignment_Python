import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ssp
import sys
import random
import math
import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

'''
    ' Author: Yidi Wang
    ' language version: Python 3.6.3
    ' All the packages above (excluding default packages) are downloaded from Laboratory for Fluorescence Dynamics, University of California, Irvine.
    ' link: http://www.lfd.uci.edu/~gohlke/pythonlibs
    ' Please update your packages to cp36-win_amd64 version if you can not run this code
    ' sklearn module is used only for building sparse matrix
    '''

def loadLabel(filename):
    binfile = open(filename, 'rb')
    line = binfile.readline()
    labellist = []
    while line:
        # eliminate fake non-empty line
        if str(line).replace(" ","") == "":
            break
        mystr = int((str(line).split()[1])[0])
        labellist.append(mystr)
        line = binfile.readline()
    return labellist

def loadData(filename):
    binfile = open(filename, 'rb')
    line = binfile.readline()
    totaldict = {}
    worddict = {}
    linecount = 0
    while line:
        # eliminate fake non-empty line
        if str(line).replace(" ","") == "":
            break
        words = str(line).split()[1:]
        mydict = {}
        for word in words:
            if word not in worddict:
                worddict[word] = len(worddict)
            if word in mydict:
                mydict[word] += 1
            else:
                mydict[word] = 1
        totaldict[linecount] = mydict
        line = binfile.readline()
        linecount += 1
    wordcount = len(worddict)
    datamatrix = np.zeros((wordcount, linecount), dtype=np.int)
    for i in range(linecount):
        curdict = totaldict.get(i)
        for j in curdict:
            numj = worddict.get(j)
            datamatrix[numj][i] = curdict.get(j)
#            print(i, "  ",j, "  ", curdict.get(j))
    return worddict, datamatrix, totaldict

def toSparseMatrix(datamatrix):
    row = []
    col = []
    values = []
    lengths = [len(datamatrix), len(datamatrix[0])]
    for i in range(0, lengths[0]):
        for j in range(0, lengths[1]):
            if datamatrix[i][j] != 0:
                row.append(i)
                col.append(j)
                values.append(datamatrix[i][j])
    c = ssp.coo_matrix((values,(row,col)), shape = (lengths[0], lengths[1]))
    return c

def saveMatrix(filename, adfilename):
    f = open("farm-ads.txt")
    ads = f.readlines()
    vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
    matrix = vectorizer.fit_transform(ads)
    sio.savemat(filename, {'ads': matrix.T})
    return

# select random fraction from dataset and return it as trainning set
def splitDataset(dataset, splitRatio):  
    trainSize = int(len(dataset) * splitRatio)  
    trainSet = []  
    copy = list(dataset)
    # 
    while len(trainSet) < trainSize:  
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return trainSet

# return a "bag of word" for each label
def summarize(separate, V):
    summarize0 = {}
    summarize1 = {}
    sum0 = 0
    sum1 = 0
    # summarize for data whose label=0
    for i in separate.get(0):
        for j in range(len(i)):
            if i[j] == 0:
                continue;
            if j not in summarize0.keys():
                summarize0[j] = i[j]
            else:
                summarize0[j] = summarize0[j] + i[j]
            sum0 += i[j]
    # summarize for data whose label=1
    for i in separate.get(1):
        for j in range(len(i)):
            if i[j] == 0:
                continue;
            if j not in summarize1.keys():
                summarize1[j] = i[j]
            else:
                summarize1[j] = summarize1[j] + i[j]
            sum1 += i[j]
    summarizep0 = {}
    summarizep1 = {}
    for i in summarize0.keys():
        summarizep0[i] = calculateProbability(i, summarize0, sum0, V, summarize0[i])
        #print(i, " ", summarizep0[i])
    for i in summarize1.keys():
        summarizep1[i] = calculateProbability(i, summarize1, sum1, V, summarize1[i])
    return [summarize0, summarize1, sum0, sum1, summarizep0, summarizep1]

# NBC start
def startNBC(train, wordcount, test):
    separatedtrain = separateByClass(train)
    predictions = getPredictions(separatedtrain, wordcount, test)
    accuracy = getAccuracy(test, predictions)
    return accuracy

# separate training dataset elements by their labels
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):  
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated  

# count(w,y) + 1
# --------------
# count(y) + (V)
def calculateProbability(index, summarizesforclass, county, V, x):
    result = math.log((x + 1) / (county + V))
    return result

def calculateClassProbabilities(summarizes, separatedtrain, V, inputVector):  
    probabilities = {}  
    for classValue in [0,1]:  
        probabilities[classValue] = 1
        index = 0
        for x in inputVector:
            if x == 0:
                index += 1
                continue
            if index in summarizes[classValue].keys():
                probabilities[classValue] += x * summarizes[classValue+4][index] #calculateProbability(index, summarizes[classValue],summarizes[(classValue + 2)], V, x)
                index += 1
            else:
                probabilities[classValue] += x * math.log(1/(summarizes[(classValue + 2)] + V))
                index += 1
    #print(probabilities[0])
    #print(probabilities[1])
        
    return probabilities

def predict(summarizes, separatedtrain, V, inputVector):  
    probabilities = calculateClassProbabilities(summarizes, separatedtrain, V, inputVector)  
    bestLabel, bestProb = None, -1  
    for classValue, probability in probabilities.items():  
        if bestLabel is None or probability > bestProb:  
            bestProb = probability  
            bestLabel = classValue  
    return bestLabel

def getPredictions(separatedtrain, V, testSet):  
    predictions = []
    #print("summarizing, time:", time.strftime('%H:%M:%S',time.localtime(time.time())))
    summarizes = summarize(separatedtrain, V)
    #print("predicting, time:", time.strftime('%H:%M:%S',time.localtime(time.time())))
    for i in range(len(testSet)):  
        #print(", time:", time.strftime('%H:%M:%S',time.localtime(time.time())))
        result = predict(summarizes, separatedtrain, V, testSet[i])  
        predictions.append(result)  
    return predictions

# calculate the accuracy
def getAccuracy(testSet, predictions):  
    correct = 0  
    for i in range(len(testSet)):  
        if testSet[i][-1] == predictions[i]:  
            correct += 1  
    return (correct/float(len(testSet))) * 100
# NBC end

# LR start
def formTrainingData(dataset):
    matrix = []
    labels = []
    for i in range(len(dataset)):
        vector = dataset[i][:]
        labels.append(vector[-1])
        vector.insert(0,1)
        matrix.append(vector[0:-1])
    return matrix, labels

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  # convert to NumPy matrix
    labelMat = np.mat(classLabels).transpose()  # convert to NumPy matrix

    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 50
    weights = np.zeros((n, 1), dtype = np.float64)

    for k in range(maxCycles):  # heavy on matrix operations
        h = sigmoid(dataMatrix * weights)  # matrix mult
        error = (labelMat - h)  # vector subtraction
        weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult
    return weights

def predict2(weights, test):
    onetest = test[:]
    onetest.insert(0,1)
    onetest = onetest[0:-1]
    testMatrix = np.mat(onetest)
    dotproduct = np.dot(testMatrix,weights)
    predict = sigmoid(dotproduct)
    label = 1 if predict>0.5 else 0
    if label == test[-1]:
        return 1
    else:
        return 0

# LR end
def trainHelper(splitRatios, dataset, N):
    train = [[[],[],[],[],[]],[[],[],[],[],[]], [[],[],[],[],[]], [[],[],[],[],[]], [[],[],[],[],[]], [[],[],[],[],[]]]
    for i in range(len(splitRatios)):
        splitRatio = splitRatios[i]
        for j in range(len(train[0])):
            train[i][j] = splitDataset(dataset, splitRatio)
    return train

def nbcHelper(splitRatios, train, wordcount, test, N):
    nbcaverageaccuracy = [None] * N
    for i in range(0, N):
        accuracy = [None] * len(train[0])
        for j in range(0, len(train[0])):
            accuracy[j] = startNBC(train[i][j], wordcount, test)
        nbcaverageaccuracy[i] = round(sum(i for i in accuracy) / len(accuracy), 2)
    return nbcaverageaccuracy

def lrHelper(train, test, N):
    lraverageaccuracy = [None] * N
    for i in range(0,N):
        accuracy = [None] * len(train[0])
        for j in range(0, len(train[0])):
            dataMat, classLabels= formTrainingData(train[i][j])
            weights = gradAscent(dataMat, classLabels)
            #print (weights)
            correct = 0
            for testnum in range(len(test)):
                correct += predict2(weights, test[testnum])
                accuracy[j] = correct / (float)(len(test)) * 100
        lraverageaccuracy[i] = round(sum(i for i in accuracy) / len(accuracy), 2)
    return lraverageaccuracy

def plotfigure(mysplitratio, myaccuracy1, myaccuracy2):
    plt.figure(figsize=(8,4))
    plt.plot(mysplitratio, myaccuracy1, 'b*',label=" Naive Bayes accuracy(%)")
    plt.plot(mysplitratio, myaccuracy1, 'r')
    plt.plot(mysplitratio, myaccuracy2, 'g*', label="Logistic Regression accuracy(%)")
    plt.plot(mysplitratio, myaccuracy2, 'y')
    plt.xlabel("split ratio")
    plt.ylabel("accuracy(%)")
    plt.show()
    return

def main():
    print("Loading data, time:", time.strftime('%H:%M:%S',time.localtime(time.time())))
    labellist = loadLabel("farm-ads-label.txt")
    worddict, datamatrix, datadict = loadData("farm-ads.txt")

    print("building sparse matrix, time:", time.strftime('%H:%M:%S',time.localtime(time.time())))
    sparsedatamatrix = toSparseMatrix(datamatrix)

    print("Saving data to .mat, time:", time.strftime('%H:%M:%S',time.localtime(time.time())))
    saveMatrix("data_matrix.mat", "farm-ads.txt")

    print("splitting training dataset, time:", time.strftime('%H:%M:%S',time.localtime(time.time())))
    splitRatios = [0.1,0.3,0.5,0.7,0.8,0.9]
    # get a complete data matrix with labels
    dataset = np.c_[sparsedatamatrix.T.toarray(), np.array(labellist)].tolist()
    test = dataset
    train = trainHelper(splitRatios, dataset, len(splitRatios))

    print("NBC start, time:", time.strftime('%H:%M:%S',time.localtime(time.time())))
    nbcaverageaccuracy = nbcHelper(splitRatios, train, sparsedatamatrix.getnnz(), test, len(splitRatios))
    for i in range(len(splitRatios)):
        print("split ratio = ",splitRatios[i],', Accuracy: %s' %(nbcaverageaccuracy[i]))
    print("NBC finished, time:", time.strftime('%H:%M:%S',time.localtime(time.time())))

    print("LR start, time:", time.strftime('%H:%M:%S', time.localtime(time.time())))
    lraverageaccuracy = lrHelper(train, test, len(splitRatios))
    for i in range(len(splitRatios)):
        print("split ratio = ",splitRatios[i],', Accuracy: %s' %(lraverageaccuracy[i]))
    print("LR finished, time:", time.strftime('%H:%M:%S', time.localtime(time.time())))

    plotfigure(splitRatios,nbcaverageaccuracy, lraverageaccuracy)

if __name__ == "__main__":
    sys.exit(main())
