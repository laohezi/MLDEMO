from math import  log

def createDataSet():
    dataSet = [[1, 1, 1,'yes'],
               [1, 1, 1,'yes'],
               [1, 0, 1,'no'],
               [0, 1, 1,'no'],
               [0, 1, 1,'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels


def calcShanonEnt(dataSet):
    numEntries = len(dataSet)
    lableCounts = {}

    for featVec in dataSet:
        currentLable = featVec[-1]
        if currentLable not in  lableCounts.keys():
            lableCounts[currentLable] = 0
        lableCounts[currentLable]  += 1
    shannonEnt = 0.0
    for key in lableCounts:
        prob = float(lableCounts[key])/numEntries
        shannonEnt -= prob *log(prob,2)
    return  shannonEnt;

def splitDataSet(dataSet,axis,value):
    retDataSet= []
    for fetVect in dataSet:
        if fetVect[axis]  == value:
            reducedFeatVect = fetVect[:axis]
            reducedFeatVect.extend(fetVect[axis+1:])
            retDataSet.append(reducedFeatVect)
    return  retDataSet

def chooseBastFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) -1
    baseEntropy = calcShanonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShanonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return  bestFeature

print(splitDataSet(createDataSet()[0],0,1))



