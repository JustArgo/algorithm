'''
	kNN步骤 k nearest neighbor
	1 n个向量  每个向量 一个类别标签
	2 一个未知类别的向量，
	3 计算未知类别的向量 和 n个向量的距离
	4 取距离最近的  n个中的 k个  
	5 k个中  哪个分类最多，则 未知类别的向量 属于该分类
'''
from numpy import *
import operator
def createDataSet():
	group = array([
				[1.0,1.1],
				[1.0,1.0],
				[0,0],
				[0,0.1]
			])
	labels = ['A','A','B','B']
	return group,labels

#此处要注意 传参数的格式为 二维 list  还是 narray  还是matrix  narray可以进行计算 list不行
def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	#inX  这个list  重复  行dataSetSize次  列 1次
	diffMat = tile(inX,(dataSetSize,1))-dataSet
	sqDiffMat = array(map(sum,diffMat**2))
	distances = sqDiffMat**0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[int(sortedDistIndicies[i])]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]
	
#从文件中 加载训练数据
def loadDataSet(trainFileName):
	fopen = open(trainFileName)
	dataSet = []
	labels = []
	for line in fopen.readlines():
		lineList = line.strip().split()
		feat = []
		for i in range(len(lineList)-1):
			feat.append(float(lineList[i]))
		labels.append(lineList[len(lineList)-1])
		dataSet.append(feat)
	return array(dataSet),labels

#从文件中加载 测试数据
def loadTestSet(testFileName):
	fopen = open(testFileName)
	dataSet = []
	for line in fopen.readlines():
		lineList = line.strip().split()
		dataSet.append(map(float,lineList))
	return dataSet

	
#trainFileName 代表要训练的 文件 最后一列为 分类标签
#testFileName  代表要分类的文件
#k 代表要使用的k个点
def knn(trainFileName,testFileName,k):
	trainDataSet,labels = loadDataSet(trainFileName)
	testDataSet = loadTestSet(testFileName)
	result = []
	for i in range(len(testDataSet)):
		result.append(classify0(array(testDataSet[i]),trainDataSet,labels,k))
	return result