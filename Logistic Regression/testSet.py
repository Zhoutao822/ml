#coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import random
"""
梯度上升
求函数f(x) = -x^2 + 4x的极大值

Parameters:
    无
Returns:
    无
"""

def Gradient_test():
    def f_prime(old_val):
        return -2 * old_val + 4
    old_val = -1
    new_val = 0
    alpha = 0.01
    presision = 0.00000001
    while abs(old_val - new_val) > presision:
        old_val = new_val
        new_val = old_val + alpha * f_prime(old_val)
    print(new_val)


"""
读取testSet的数据，绘制图像
"""

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        curLine = line.strip().split()
        dataMat.append([1.0, float(curLine[0]), float(curLine[1])])
        labelMat.append(int(curLine[-1]))
    return dataMat, labelMat

def plotDataSet(weights):
    dataArr, labelMat = loadDataSet()
    dataMat = np.array(dataArr)
    n = np.shape(dataMat)[0]
    x1 = []; y1 = []
    x2 = []; y2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            x1.append(dataMat[i][1]); y1.append(dataMat[i][2])
        else:
            x2.append(dataMat[i][1]); y2.append(dataMat[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, s = 20, c = 'red', marker = 's', alpha = .5)
    ax.scatter(x2, y2, s = 20, c = 'yellow', alpha = .5)
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('DataSet')
    plt.xlabel('x1'); plt.ylabel('x2')
    plt.show()

"""
使用sigmoid，梯度上升
"""

def sigmoid(inX):
    return 1.0 / (1.0 + np.exp(-inX))

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

"""
采取随机梯度上升，减少运算次数
"""

def randGradAscent(dataMatrix, labelMat, maxCycles = 150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(maxCycles):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4.0 / (1 + j + i) + 0.01
            randomIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randomIndex] * weights))
            error = labelMat[randomIndex] - h
            weights += alpha * error * dataMatrix[randomIndex]
            del(dataIndex[randomIndex])
    return weights

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    plotDataSet(gradAscent(dataMat, labelMat))
    plotDataSet(randGradAscent(np.array(dataMat), labelMat))
