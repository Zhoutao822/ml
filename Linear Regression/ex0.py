# -*- coding:utf-8 -*-
#线性回归

import matplotlib.pyplot as plt
import numpy as np


"""
    函数说明:加载数据
    Parameters:
        fileName - 文件名
    Returns:
        xArr - x数据集
        yArr - y数据集
"""
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    xArr = []; yArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr

"""
    函数说明:计算回归系数w
    Parameters:
        xArr - x数据集
        yArr - y数据集
    Returns:
        ws - 回归系数    
"""
def standRegres(xArr, yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print('The mat can not be reversed')
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

"""
    函数说明:绘制数据集
    Parameters:
        无
    Returns:
        无
"""
def plotDataSet():
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr, yArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    print(np.corrcoef((xMat * ws).T, yMat))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xCopy[:, 1], yHat, c = 'red') #划线

    ax.scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue', alpha = .5) #描点
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

if __name__ == '__main__':
    plotDataSet()