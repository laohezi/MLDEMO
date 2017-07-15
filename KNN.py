from numpy import *
import operator
import matplotlib.pyplot as plt


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, lables, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndiacies = distances.argsort()
    classCount = {}
    for i in range(k):
        # 选取距离最小的k个点
        votelabel = lables[sortedDistIndiacies[i]]
        classCount[votelabel] = classCount.get(votelabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 从文件读取数据
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # get the number of lines in the file
    returnMat = zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 归一化公式：newValue = (oldValue - min)/(max - min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals

    normalDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normalDataSet = dataSet - tile(minVals, (m, 1))
    normalDataSet = normalDataSet / tile(ranges, (m, 1))
    return normalDataSet, ranges, minVals


def getDatingTestSet2Mat():
    datingDataMat, dataLabels = file2matrix("D:\文档\machinelearninginaction\Ch02\datingTestSet2.txt")
    return datingDataMat, dataLabels


def plot():
    datingDataMat, dataLabels = getDatingTestSet2Mat()

    fig = plt.figure()

    ax = fig.add_subplot(111)
    # x:x坐标的值
    # y:y坐标
    # s:常量或者数组 表示圆点的size
    # s:常量或者数组 表示圆点的颜色
    ax.scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], s=15.0 * array(dataLabels), c=15.0 * array(dataLabels))
    plt.show()
def datingClassTest():
    hoRatio = 0.10 #测试数据集的比率
    datingLabels = getDatingTestSet2Mat()[1]
    normMat,ranges,minVals = autoNorm(getDatingTestSet2Mat()[0])
    m= normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d,the real answer is %d" %(classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):errorCount +=1.0
    print("the total error rate is : %f"%(errorCount/float(numTestVecs)))

datingClassTest()




