#coding=utf-8
import re
import random
import numpy as np


def textParse(data):
    listOfWords = re.split(r'\W*', data)
    return [tok.lower() for tok in listOfWords if len(tok) > 2]

def createVocabList(dataSet):
    vocabList = set([])
    for line in dataSet:
        vocabList = vocabList | set(line)
    return list(vocabList)

def listOfWords2Vec(vocabList, input):
    returnVec = [0] * len(vocabList)
    for word in input:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def setOfWords2Vec(vocabList, input):
    returnVec = [0] * len(vocabList)
    for word in input:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec

def trainNB(dataMat, labels):
    numDocs = len(dataMat)
    numWords = len(dataMat[0])
    p1 = sum(labels) / float(numDocs)
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numDocs):
        if labels[i] == 1:
            p1Num += dataMat[i]
            p1Denom += sum(dataMat[i])
        else:
            p0Num += dataMat[i]
            p0Denom += sum(dataMat[i])
    p1Vec = np.log(p1Num / p1Denom)
    p0Vec = np.log(p0Num / p0Denom)
    return p0Vec, p1Vec, p1

def classifyNB(testInput, p0Vec, p1Vec, p1):
    pt1 = sum(testInput * p1Vec) + np.log(p1)
    pt0 = sum(testInput * p0Vec) + np.log(1 - p1)
    print('p1 is : %.2f and p0 is : %.2f' % (pt1, pt0))
    if pt0 < pt1:
        return 1
    else:
        return 0

def spanTest():
    docList = []; labels = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%s.txt' % i, 'r').read())
        docList.append(wordList)
        labels.append(1)
        wordList = textParse(open('email/ham/%s.txt' % i, 'r').read())
        docList.append(wordList)
        labels.append(0)
    vocabList = createVocabList(docList)
    print("vocabList:")
    print(vocabList)
    indexList = list(range(50)); testIndex = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(indexList)))
        testIndex.append(indexList[randIndex])
        del(indexList[randIndex])
    trainMat = []; trainLabels = []
    for index in indexList:
        trainMat.append(listOfWords2Vec(vocabList,docList[index]))
        trainLabels.append(labels[index])
    p0v, p1v, ps = trainNB(trainMat, trainLabels)
    errorCount = 0
    for index in testIndex:
        wordVec = listOfWords2Vec(vocabList, docList[index])
        if classifyNB(np.array(wordVec), p0v, p1v, ps) != labels[index]:
            errorCount += 1
            print('错误测试集为：', docList[index])
    print('错误率：%.2f%%' % (float(errorCount) / len(testIndex) * 100))

if __name__ == '__main__':
    spanTest()