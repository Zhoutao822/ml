#coding=utf-8
import numpy as np
import random
from sklearn.linear_model import LogisticRegression

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def classify(inX, weights):
    prob = sigmoid(sum(inX * weights))
    return 1.0 if prob > 0.5 else 0.0

def gradAscent(dataMat, labelMat):
    dataMatrix = np.mat(dataMat)
    labelMatrix = np.mat(labelMat).T
    alpha = 0.01
    maxCycles = 1000
    m, n = np.shape(dataMatrix)
    weights = np.ones((n, 1))
    for i in range(maxCycles):
        tmp = sigmoid(dataMatrix * weights)
        error = labelMatrix - tmp
        weights += alpha * dataMatrix.T * error
    return weights.flatten()


def randGradAscent(dataMatrix, labelMat, maxCycles = 150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(maxCycles):
        dataIndex = list(range(m))
        for i in range(m):
            randomIndex = int(random.uniform(0, len(dataIndex)))
            alpha = 4.0 / (i + j + 1) + 0.01
            h = sigmoid(sum(dataMatrix[randomIndex] * weights))
            error = labelMat[randomIndex] - h
            weights += alpha * error * dataMatrix[randomIndex]
            del(dataIndex[randomIndex])
    return weights

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        curLine = line.strip().split()
        lineArr = []
        for i in range(len(curLine) - 1):
            lineArr.append(float(curLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(curLine[-1]))
    
    weights = gradAscent(np.array(trainingSet), trainingLabels)
    errorCount = 0; num = 0.0
    for line in frTest.readlines():
        num += 1
        curLine = line.strip().split()
        lineArr = []
        for i in range(len(curLine) - 1):
            lineArr.append(float(curLine[i]))
        if int(classify(np.array(lineArr), weights)) != int(curLine[-1]):
            errorCount += 1
    errorRate = float(errorCount/num) * 100
    print("测试集错误率为: %.2f%%" % errorRate)

def colicSklearn():
    frTrain = open('horseColicTraining.txt')                                       
    frTest = open('horseColicTest.txt')                                               
    trainingSet = []; trainingLabels = []
    testSet = []; testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))
    classifier = LogisticRegression(solver = 'sag', max_iter = 5000).fit(trainingSet, trainingLabels)
    accuracy = classifier.score(testSet, testLabels) * 100
    print('正确率:%f%%' % accuracy)

if __name__ == '__main__':
    colicTest()
    colicSklearn()
