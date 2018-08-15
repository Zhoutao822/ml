# -*- coding:utf-8 -*-
# 岭回归(Ridge Regression)和逐步线性回归

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(fileName):
    featNum = len(open(fileName).readline().split('\t')) - 1
    xArr = []; yArr = []
    fr = open(fileName)
    for line in fr.readlines():
        tmp = []
        curLine = line.strip().split('\t')
        for i in range(featNum):
            tmp.append(float(curLine[i]))
        xArr.append(tmp)
        yArr.append(float(curLine[-1]))
    return xArr, yArr

def ridgeRegres(xMat, yMat, lam = 0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print('The mat can not be reversed')
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    yMean = np.mean(yMat, axis = 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, axis = 0)
    xVar = np.var(xMat, axis = 0)
    xMat = (xMat - xMeans) / xVar
    num = 30
    wMat = np.zeros((num, np.shape(xMat)[1]))
    for i in range(num):
        tmp = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = tmp.T
    return wMat

def plotMat():
    font = FontProperties(None, size=14)
    abX, abY = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    ax_title_text = ax.set_title(u'log(lambada)与回归系数的关系', FontProperties = font)
    ax_xlabel_text = ax.set_xlabel(u'log(lambada)', FontProperties = font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties = font)
    plt.setp(ax_title_text, size = 20, weight = 'bold', color = 'red')
    plt.setp(ax_xlabel_text, size = 10, weight = 'bold', color = 'black')
    plt.setp(ax_ylabel_text, size = 10, weight = 'bold', color = 'black')
    plt.show()

def regularize(xMat, yMat):
    inxMat = xMat.copy()
    inyMat = yMat.copy()
    yMean = np.mean(inyMat, 0)
    inyMat = yMat - yMean
    inxMeans = np.mean(inxMat, 0)
    inxVar = np.var(inxMat, 0)
    inxMat = (inxMat - inxMeans) / inxVar
    return inxMat, inyMat

def rssError(yMat, yTest):
    return ((yMat.A - yTest.A)**2).sum()

def stageWise(xArr, yArr, esp = 0.01, numIt = 100):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xMat, yMat = regularize(xMat, yMat)
    n = np.shape(xMat)[1]
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsMax = ws.copy()
    wsTest = ws.copy()

    for i in range(numIt):
        lowestError = float('inf')
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += sign * esp
                yTest = xMat * wsTest
                rssE = rssError(yMat, yTest)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat 

def plotstageWiseMat():
    font = FontProperties(None, size=14)
    xArr, yArr = loadDataSet('abalone.txt')
    returnMat = stageWise(xArr, yArr, 0.005, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(returnMat)    
    ax_title_text = ax.set_title(u'前向逐步回归:迭代次数与回归系数的关系', FontProperties = font)
    ax_xlabel_text = ax.set_xlabel(u'迭代次数', FontProperties = font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties = font)
    plt.setp(ax_title_text, size = 15, weight = 'bold', color = 'red')
    plt.setp(ax_xlabel_text, size = 10, weight = 'bold', color = 'black')
    plt.setp(ax_ylabel_text, size = 10, weight = 'bold', color = 'black')
    plt.show()

if __name__ == '__main__':
    plotMat()
    plotstageWiseMat()