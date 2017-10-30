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

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape(0)
	diffMat = tile(inX,(dataSetSize,1))-dataSet
	sqDiffMat = diffMat**2
	distances = sqDiffMat**0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]