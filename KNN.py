from numpy import  *
import  operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return  group,labels

def classify0(inX, dataSet, lables, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) -dataSet
    sqDiffMat = diffMat **2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances **0.5
    sortedDistIndiacies = distances.argsort()
    classCount = {}
    for i in range(k):
        #选取距离最小的k个点
        votelabel = lables[sortedDistIndiacies[i]]
        classCount[votelabel] = classCount.get(votelabel,0)+1

    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
def file2matrix(fileName):
    fr = open(fileName)
    arrayOfLines = fr.readline()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0

    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('/t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return  returnMat,classLabelVector



