# -*- coding:utf-8 -*-
# BeautifulSoup爬取网页信息和使用信息进行交易价格预测

from scrapeLego import *
import numpy as np
import abalone as abalone
import ex0 as ex0
import random

def useStandRegres():
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    dataNum, featNum = np.shape(lgX)
    lgX1 = np.mat(np.ones((dataNum, featNum + 1)))
    lgX1[:, 1:5] = np.mat(lgX)
    ws = ex0.standRegres(lgX1, lgY)
    print('%f% + f * 年份% + f * 部件数量% + f * 是否为全新% + f * 原价' % (ws[0], ws[1], ws[2], ws[3], ws[4]))

def rssError(yArr,yHatArr):
    return ((yArr - yHatArr)**2).sum()

"""
    交叉验证岭回归
"""
def crossValidation(xArr, yArr, numVal = 10):
    m = len(yArr)
    indexList = list(range(m))
    errorMat = np.zeros((numVal, 30))
    for i in range(numVal):
        trainX = []; trainY = []
        testX = []; testY = []
        random.shuffle(indexList)

        for j in range(m):
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = abalone.ridgeTest(trainX, trainY)
        for k in range(30):
            matTestX = np.mat(testX); matTrainX = np.mat(trainX)
            meanTrain = np.mean(matTrainX, 0)
            varTrainX = np.var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrainX
            yEst = matTestX * np.mat(wMat[k, :]).T + np.mean(trainY)
            errorMat[i, k] = rssError(yEst.T.A, np.array(testY))
    meanErrors = np.mean(errorMat, 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    meanX = np.mean(xMat, 0); varX = np.var(xMat, 0)
    unReg = bestWeights / varX
    print('%f% + f * 年份% + f * 部件数量% + f * 是否为全新% + f * 原价' % ((-1 * np.sum(np.multiply(meanX, unReg)) + np.mean(yMat)), unReg[0, 0], unReg[0, 1], unReg[0, 2], unReg[0, 3]))


if __name__ == '__main__':
    useStandRegres()
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    crossValidation(lgX, lgY)
