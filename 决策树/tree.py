'''
	决策树的步骤
	1 n个数据  n行 m列  最后一列代表 具体分类
	2 关键是熵的计算 -p(x)log2.p(x) 计算熵的时候，只算  最后一列的分类
	3 循环特征 按特征划分之后 计算熵  比较信息增益大的 代表 划分之后 排序更有序 先取特征好的 
'''
from math import log
import treePlotter #引入决策树 辅助处理模块

#从文件加载数据
def loadDataSet(fileName):
	fopen = open(fileName)
	dataArr = []
	labelSet = ['no surfacing','flippers']
	for line in fopen.readlines():
		list = []
		textList = line.strip().split()
		#labelSet.add(textList[len(textList)-1])
		for text in textList:
			list.append(text)
		dataArr.append(list)
	return dataArr,labelSet

def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		shannonEnt -= prob * log(prob,2)
	return shannonEnt

#按照给定特征划分数据集	
def splitDataSet(dataSet,axis,value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0])-1
	bestEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0
	bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
		infoGain = bestEntropy - newEntropy
		if(infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature
	
#如果划分到最后 特征用完 依然不能将 类别区分开 则用一下方法
import operator
def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
	sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]
	
#创建树的函数代码
def createTree(dataSet,labels):
	classList = [example[-1] for example in dataSet]
	if classList.count(classList[0]) == len(classList): #如果类别完全相同 则直接返回第一个类别
		return classList[0]
	if len(dataSet[0])==1:
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
	return myTree
	
#使用决策树的分类函数
def classify(inputTree,featLabels,testVec):
	firstStr = list(inputTree.keys())[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__=='dict':
				classLabel = classify(secondDict[key],featLabels,testVec)
			else:
				classLabel = secondDict[key]
	return classLabel
	

def dtree(fileName):
	myDat,labels = loadDataSet(fileName) #返回数据二维数组，数据的标签，直接用最后一列最为标签
	myTree = treePlotter.retrieveTree(0)
	print(classify(myTree,labels,[1,0]))
	
#def dtree(fileName):
#实际分类代码

#myTree = treePlotter.retrieveTree(0)
#trees.classify(myTree,labels,[1,0])
#trees.classify(myTree,labels,[1,1])

