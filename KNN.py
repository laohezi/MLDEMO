from numpy import *
import operator
import matplotlib.pyplot as plt
import  os


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# knn分类
def classify0(inX, dataSet, lables, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndiacies = distances.argsort()  # 按照距离大小的顺序排列
    classCount = {}
    for i in range(k):
        # 选取距离最小的k个点
        votelabel = lables[sortedDistIndiacies[i]]
        # 如果该标签中标后则再对其中标次数加一，最后中标次数最多的标签则视为该测试对象的标签
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
    hoRatio = 0.10  # 测试数据集的比率
    datingLabels = getDatingTestSet2Mat()[1]
    normMat, ranges, minVals = autoNorm(getDatingTestSet2Mat()[0])
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d,the real answer is %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is : %f" % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['dislike', 'ok', 'love']
    percentTats = float(input("玩游戏的时间百分比"))
    ffMiles = float(input("每年的飞行距离"))
    iceCream = float(input("吃冰淇淋的数量"))

    datingMat, datingLabels = getDatingTestSet2Mat();
    normMat, ranges, minVals = autoNorm(datingMat)

    inArr = array([ffMiles, percentTats, iceCream])

    result = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)

    print("你会", resultList[result - 1], "这个人")
    classifyPerson()


def image2Vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFilelist = os.listdir("D:\文档\machinelearninginaction\Ch02\drainingDigits")
    m = len(trainingFilelist)
    trainingMat = zeros((m,1024))
    for  i in range(m):
        fileNameStr = trainingFilelist[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i:] = image2Vector('D:\文档\machinelearninginaction\Ch02\drainingDigits/%s'%fileNameStr)
    testFileList =os.listdir('D:\文档\machinelearninginaction\Ch02\destDigits')
    errorCount = 0.0
    mTest = len(testFileList)

    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = image2Vector('D:\文档\machinelearninginaction\Ch02\destDigits/%s' %fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier come back with: %d,the real answer id :%d" %(classifierResult,classNumStr))
        if (classifierResult != classNumStr):errorCount +=1.0
    print("/n the total number of error is:%d"%errorCount)
    print("/n the total error ratio is : %f" %(errorCount/float(mTest)))

handwritingClassTest()


