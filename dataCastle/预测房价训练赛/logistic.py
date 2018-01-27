'''
	本算法用于解决二值分类问题
	1 给定 n个特征向量 和 其对应的 分类 labels  假设特征为2个
	2 找出回归系数 x0 x1 x2 由于此处特征为2个 所以找3个回归系数 x0类似于 直线的截点
	3 由于是二值问题 所以 用 sigmoid 阶跃函数 来修正 回归系数
'''
from numpy import *
import re
logSigmoid = False
#加载数据
def loadDataSet(trainFileName):
	dataMat = []
	labelMat = []
	fr = open(trainFileName)
	for line in fr.readlines():
		lineArr = line.strip().split()
		dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
		labelMat.append([int(lineArr[2])])
	return dataMat,labelMat

#本方法针对那些不是 全部数值型的数据,并且有多个特征的 此处外加对dataCastle的特殊处理
def dealDataSet(trainFileName,separator):
	matcher = re.compile(r'^[-+]?[0-9]+\.?[0-9]*$')
	dataMat = []
	labelMat = []
	fr = open(trainFileName)
	for line in fr.readlines():
		lineArr = line.strip().split(separator)
		list = []
		list.append(1.0)
		for i in range(len(lineArr)):
			if i!=1:
				list.append(lineArr[i])
			else:
				labelMat.append([int(lineArr[1])])
		dataMat.append(list)
	for j in range(len(dataMat[0])):
		if matcher.match(str(dataMat[0][j]))==None:
			colDict = {}
			value = 1
			for i in range(len(dataMat)):
				if dataMat[i][j] not in colDict:
					colDict[dataMat[i][j]] = value
					value += 1
			for i in range(len(dataMat)):
				if dataMat[i][j] in colDict:
					dataMat[i][j] = colDict[dataMat[i][j]]
	for i in range(len(dataMat)):
		for j in range(len(dataMat[i])):
			dataMat[i][j] = float(dataMat[i][j])
	return dataMat,labelMat
	
#sigmoid 阶跃函数	
def sigmoid(inX):
	if logSigmoid == True:
		print('log sigmoid start:')
		print('in is %s and out is %s' % (inX,1.0/(1+exp(-inX))))
		print('log sigmoid end:')
	return 1.0/(1+exp(-inX))

#梯度上升算法
def gradAscent(dataMatIn, classLabels):
	dataMatrix = mat(dataMatIn)
	labelMat = mat(classLabels)
	m,n = shape(dataMatrix)
	alpha = 0.001
	maxCycles = 500
	weights = ones((n,1)) * 0.1
	for k in range(maxCycles):
		h = sigmoid(dataMatrix * weights)
		#print(labelMat)
		#print(h)
		error = (labelMat - h)
		#print(h)
		#print(error)
		#print(dataMatrix.transpose())
		#print(alpha * dataMatrix.transpose() * error)
		weights = weights + alpha * dataMatrix.transpose() * error
	return weights
	
#dataArr,labelMat = loadDataSet()
#weights = gradAscent(dataArr,labelMat)
#print(weights)

#归一化数值 传list 进来
def autoNorm(dataSet):
	dataSet = array(dataSet)
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals-minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet-tile(minVals,(m,1))
	normDataSet = normDataSet/tile(ranges,(m,1))
	return normDataSet.tolist()

#画出数据集 和 logistic 回归的最佳拟合直线的函数
def plotBestFit(wei):
	import matplotlib.pyplot as plt
	weights = wei
	dataMat,labelMat = loadDataSet()
	dataArr = array(dataMat)
	n = shape(dataArr)[0]
	xcord1 = []
	ycord1 = []
	xcord2 = []
	ycord2 = []
	for i in range(n):
		if int(labelMat[i][0]) == 1:
			xcord1.append(dataArr[i,1])
			ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i,1])
			ycord2.append(dataArr[i,2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
	ax.scatter(xcord2,ycord2,s=30,c='green')
	x = arange(-8.0,8.0,0.1)
	print(weights)
	y = (-weights[0]-weights[1]*x)/weights[2]
	ax.plot(x,y)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()
	
#随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
	m,n = shape(dataMatrix)
	alpha = 0.5
	weights = ones(n)
	#weights = array([1.0,-1.0,1.0])
	for i in range(m):
		h = sigmoid(sum(dataMatrix[i]*weights))
		error = classLabels[i] - h
		weights = weights + alpha * error * dataMatrix[i] 
	return weights
	
#改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
	m,n = shape(dataMatrix)
	weights = ones(n)
	for j in range(numIter):
		dataIndex = range(m)
		for i in range(m):
			alpha = 4/(1.0+j+i)+0.01
			randIndex = int(random.uniform(0,len(dataIndex)))
			h = sigmoid(sum(dataMatrix[randIndex]*weights))
			error = classLabels[randIndex] - h
			weights = weights + alpha * error * dataMatrix[randIndex]
			del(dataIndex[randIndex])
	return weights

#加载测试集数据
def loadTestSet(testFileName,separator):
	matcher = re.compile(r'^[-+]?[0-9]+\.?[0-9]*$')
	dataMat = []
	fr = open(testFileName)
	for line in fr.readlines():
		lineArr = line.strip().split(separator)
		list = []
		list.append(1.0)
		for i in range(len(lineArr)):
			list.append(lineArr[i])
		dataMat.append(list)
	for j in range(len(dataMat[0])):
		if matcher.match(str(dataMat[0][j]))==None:
			colDict = {}
			value = 1
			for i in range(len(dataMat)):
				if dataMat[i][j] not in colDict:
					colDict[dataMat[i][j]] = value
					value += 1
			for i in range(len(dataMat)):
				if dataMat[i][j] in colDict:
					dataMat[i][j] = colDict[dataMat[i][j]]
	for i in range(len(dataMat)):
		for j in range(len(dataMat[i])):
			dataMat[i][j] = float(dataMat[i][j])
	return dataMat

#检验权重
def testWeights(testDataArr,testLabelMat,weights):
	estimateMat = testDataArr*weights;
	correctCount = 0
	for i in range(len(testLabelMat)):
		if testLabelMat[i,0]==estimateMat[i]:
			correctCount += 1
	return correctCount/len(testLabelMat)
	

#验证分类是否正确
def classifyVector(inX, weights):
	prob = sigmoid(sum(inX*weights))
	if prob > 0.5:
		return 1.0
	else:
		return 0.0
	
#将结果写到文件中
def writeToFile(fileName,testDataArr,weights):
	output = open(fileName, 'w')
	for i in range(len(testDataArr)):
		output.write(str(int(classifyVector(testDataArr[i],weights)))+'\n')
	output.close()
	
trainFileName = "pfm_train.csv";
testFileName = "pfm_test.csv";
dataArr,labelMat = dealDataSet(trainFileName,',');
#dataArr = autoNorm(dataArr)
weights = stocGradAscent0(dataArr,labelMat);

#验证得到的 回归系数矩阵
testDataArr = loadTestSet(testFileName,',');
#testDataArr = autoNorm(testDataArr)
writeToFile('sample.csv',testDataArr,weights)

#print(weights)
#plotBestFit(weights)

