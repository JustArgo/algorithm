'''
	
		1 给定数据集  treeNum = 50 代表要生成50棵树，minData=3 最少3行  minFeat=10最少10个特征 树为字典 [{'index':0,'cmp':'gt','threshVal':0.5,cals:'1'}]
		2 循环每一棵树，进行分类
	
'''	
from numpy import *
import math
import random


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
		shannonEnt -= prob * math.log(prob,2)
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
	
#创建树的函数代码 这里dataSet是最后一列包含分类  labels代表的是每个列的名称
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
	#print('inputTree',inputTree)
	#print('featLabels',featLabels)
	firstStr = list(inputTree.keys())[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	classLabel = ''
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
	

#加载训练数据
def loadTrainDataSet(trainFileName,separator='\t'):
	fopen = open(trainFileName)
	lineArr = []
	labelArr = []
	lines = fopen.readlines()
	labelArr = lines[0].strip().split(separator)##文件第一行为每一列特征的名称 不包括 最后一列
	for i in range(1,len(lines)):
		lineList = lines[i].strip().split(separator)
		featArr = []
		for i in range(len(lineList)):
			featArr.append(float(lineList[i]))
		lineArr.append(featArr)
	return lineArr,labelArr

#加载测试数据
def loadTestDataSet(testFileName,separator='\t'):
	fopen = open(testFileName)
	lineArr = []
	for line in fopen.readlines():
		lineList = line.strip().split(separator)
		lineArr.append(list(map(float,lineList)))
	return lineArr	
	
	
#随机森林算法   每个特征的列,都不能太复杂
def randomForest(trainFileName='trainSet.txt',testFileName='testSet.txt',separator='\t',minData=3,minFeat=2,treeNum=10):
		
	dataArr,labels = loadTrainDataSet(trainFileName,separator);
	featCount = len(dataArr[0])
	#print('labels',labels)
	#print('dataArr[0]',dataArr[0])
	treeList = []
	labelList = []
	for index in range(treeNum):	
		subData = []
		subLabel = []
		#i得到的是代表要生成多少行
		i = random.randint(minData,len(dataArr)-1)
		#j得到的是代表要生成多少列
		j = random.randint(minFeat,len(dataArr[0])-1)
		lineSet = set()
		featSet = set()
		lineSet.clear()
		featSet.clear()
		while len(lineSet) < i:
			lineSet.add(random.randint(0,len(dataArr)-1))
		while len(featSet) < j:
			featSet.add(random.randint(0,len(dataArr[0])-2))
		
		lineIndexList = list(lineSet)
		featIndexList = list(featSet)
		lineIndexList.sort()
		featIndexList.sort()
		
		for lineIndex in lineIndexList:
			lineList = []
			for featIndex in featIndexList:
				lineList.append(dataArr[lineIndex][featIndex])
			lineList.append(dataArr[lineIndex][featCount-1])
			subData.append(lineList)
		for featIndex in featIndexList:
			subLabel.append(labels[featIndex])
		#生成树
		#print(subData,subLabel)
		#exit()
		#print(subLabel)
		labelList.append(subLabel[:])
		#这一步创建完树之后，subLabel就没有了
		tree = createTree(subData,subLabel)
		#print(subLabel)
		treeList.append(tree)
		
	
	testDataArr = loadTestDataSet(testFileName,separator)
	testLabelList = []
	for testVector in testDataArr:
		clasDict = {}
		for i in range(len(treeList)):
			#print(testVector)
			#print(classify(treeList[i],labelList[i],testVector))
			if type(treeList[i]).__name__!='dict':
				continue
			clasVal = classify(treeList[i],labelList[i],testVector)
			#print(clasVal)
			if clasVal in clasDict.keys():
				clasDict[clasVal] += 1
			else:
				clasDict[clasVal] = 1
		#print(clasDict)
		testLabelList.append(sorted(clasDict.iteritems(),key=lambda p:p[1],reverse=True)[0][0])
		#testLabelList.append(clasDict)
	return testLabelList