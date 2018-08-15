#coding=utf-8
import numpy as np

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]                                                                   #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec

def createVocabList(dataSet):
    vocabList = set([])
    for line in dataSet:
        vocabList = vocabList | set(line)
    return list(vocabList)

def words2Vec(vocabList, input):
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


if __name__ == '__main__':
    dataSet, labels = loadDataSet()
    vocabList = createVocabList(dataSet)
    trainMat = []
    for data in dataSet:
        trainMat.append(words2Vec(vocabList, data))
    p0vec, p1vec, p1 = trainNB(trainMat, labels)
    print(vocabList)
    print('p0vec:')
    print(p0vec)
    print('p1vec:')
    print(p1vec)
    print('p1 is %.2f' % p1)

    testWords = ['cute', 'I', 'him', 'so', 'stupid', 'stupid', 'him']
    testVec = words2Vec(vocabList, testWords)
    result = classifyNB(np.array(testVec), p0vec, p1vec, p1)
    print(testWords)
    if result == 1:
        print('测试文字中包含侮辱性词汇较多')
    else:
        print('测试文字中包含侮辱性词汇较少或没有')
    