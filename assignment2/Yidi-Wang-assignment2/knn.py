import numpy as np
import struct
import sys
import operator
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

'''
' Author: Yidi Wang
' language version: Python 3.6.3
' All the packages above (excluding default packages) are downloaded from Laboratory for Fluorescence Dynamics, University of California, Irvine.
' link: http://www.lfd.uci.edu/~gohlke/pythonlibs
' Please update your packages to cp36-win_amd64 version if you can not run this code
'''

def loadImage(filename):
    binfile = open(filename, 'rb')
    buf = binfile.read()

    index = 0
    magic, numImages, numRows, numColumns = struct.unpack_from('>IIII',buf,index)
    index += struct.calcsize('>IIII')
    bits = numImages * numRows * numColumns
    bitsString = '>' + str(bits) + 'B'
    imgs = struct.unpack_from(bitsString, buf, index)
    binfile.close()
     
    imgs = np.reshape(imgs,[numImages, numRows * numColumns])
    return imgs

def loadLabel(filename):
    binfile = open(filename, 'rb')
    buf = binfile.read()

    index = 0
    magic, numLabels = struct.unpack_from('>II',buf,index)
    index += struct.calcsize('>II')
    bits = numLabels
    bitsString = '>' + str(bits) + 'B'
    labels = struct.unpack_from(bitsString, buf, index)
    binfile.close()
     
    labels = np.reshape(labels,[numLabels])
    return labels
    
def calculatDis(test, train):
    size = train.shape[0]
    diffbetween = np.tile(test, (size, 1)) - train
    diffsquaremat = diffbetween**2
    distances = diffsquaremat.sum(axis=1)
    sorteddistances = distances.argsort()
    return sorteddistances

def classify(sorteddistances, labels, k):
    count = {}
    for i in range(k):
        label = labels[sorteddistances[i]]
        count[label] = count.get(label, 0) + 1
    sortedcount = sorted(count.items(), key = operator.itemgetter(1), reverse = True)
    return sortedcount[0][0]

def plotfigure(myk, myaccuracy):
    plt.figure(figsize=(8,4))
    plt.plot(myk, myaccuracy, 'b*',label="accuracy(%)")
    plt.plot(myk, myaccuracy, 'r')
    plt.xlabel("value of k")
    plt.ylabel("accuracy(%)")
    plt.show()
    return

def main():
    # load data
    print("strat loading data, time:", time.strftime('%H:%M:%S',time.localtime(time.time())))
    train = loadImage('train-images.idx3-ubyte')
    trainlabels = loadLabel("train-labels.idx1-ubyte")
    test = loadImage("t10k-images.idx3-ubyte")
    testlabels = loadLabel("t10k-labels.idx1-ubyte")

    '''
    for i in range(10):
        print(np.reshape(train[i,:],[28, 28]))
        print(trainlabels[i])
    '''
    # to reduce calculating time, do dimension reduction (PCA) on both training and test data
    print("loading finished, start DR, time:", time.strftime('%H:%M:%S',time.localtime(time.time())))
    dimension = 20  # target dimension
    mypca = PCA(dimension)
    drtrain = mypca.fit_transform(train)
    drtest = mypca.transform(test)
    
    # use knn to classify and calculate the accuracy
    print("DR finished, start calculating, time:", time.strftime('%H:%M:%S',time.localtime(time.time())))
    total = testlabels.__len__()
    allk = {1,3,5,10,30,50,70,80,90,100}
    correct = {}.fromkeys(allk, 0)
    i = 0
    for atest in drtest:
#        if i % 1000 == 0 and i != 0:
#            print(i," finished, accuracy ", round(correct / i * 100, 2), " time:", time.strftime('%H:%M:%S',time.localtime(time.time())))
        # KNN classification begin:
        sorteddistances = calculatDis(atest, drtrain)
        for k in allk:
            testresult = classify(sorteddistances , trainlabels, k)
#           print(testresult," ", testlabels[i])
            if testresult == testlabels[i]:
                correct[k] += 1
        i += 1
        # KNN classification end
        
    # print the accuracy:
    for k in range(101):
        if k in allk:
            output = round(correct[k] / total * 100, 2)
            print("for k = ",k," accuracy:", output, "%")
    print("Finished, time:", time.strftime('%H:%M:%S',time.localtime(time.time())))

    # plot the accuracy figure
    myk = []
    myaccuracy = []
    for k in range(101):
        if k in allk:
            myk.append(k)
            myaccuracy.append(round(correct[k] / total * 100, 2))
    plotfigure(myk, myaccuracy)
    
if __name__ == "__main__":
    sys.exit(main())
