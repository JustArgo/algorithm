'''
	本算法用于解决二值分类问题
	1 给定 n个特征向量 和 其对应的 分类 labels  假设特征为2个
	2 找出回归系数 x0 x1 x2 由于此处特征为2个 所以找3个回归系数 x0类似于 直线的截点
	3 由于是二值问题 所以 用 sigmoid 阶跃函数 来修正 回归系数
'''
from numpy import *
logSigmoid = False
#加载数据
def loadDataSet():
	dataMat = []
	labelMat = []
	fr = open('testSet2.txt')
	for line in fr.readlines():
		lineArr = line.strip().split()
		dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
		labelMat.append([int(lineArr[2])])
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
		
#加载训练数据
def loadTrainDataSet(trainFileName,separator='\t'):
	dataMat = []
	labelMat = []
	fr = open(trainFileName)
	for line in fr.readlines():
		lineArr = line.strip().split(separator)
		lineList = [1.0]
		for val in lineArr:
			lineList.append(float(val))
		labelMat.append([int(lineList.pop())])
		dataMat.append(lineList)
		
	return dataMat,labelMat
	
#加载测试数据
def loadTestDataSet(testFileName,separator='\t'):
	dataMat = []
	fr = open(testFileName)
	for line in fr.readlines():
		lineArr = line.strip().split(separator)
		lineList = [1.0]
		for val in lineArr:
			lineList.append(float(val))
		dataMat.append(lineList)
		
	return dataMat
			
#归一化计算
def uniform(yList):
	f = lambda x:1 if x>0.5 else 0
	return map(f,yList)
	
#logistics回归 采用普通的随机梯度上升算法
def logis1(trainFileName='trainSet.txt',testFileName='testSet.txt',separator='\t'):
	dataArr,labelMat = loadTrainDataSet(trainFileName,separator)
	weights = stocGradAscent0(dataArr,labelMat)
	testDataArr = loadTestDataSet(testFileName,separator)
	return uniform(sigmoid(sum(testDataArr*weights,axis=1)))
	
	
	
#logistics回归 采用改进的随机梯度上升算法
def logis2(trainFileName='trainSet.txt',testFileName='testSet.txt',numIter=150,separator='\t'):
	dataArr,labelMat = loadTrainDataSet(trainFileName,separator)
	weights = stocGradAscent1(dataArr,labelMat,numIter)
	testDataArr = loadTestDataSet(testFileName,separator)
	return uniform(sigmoid(sum(testDataArr*weights,axis=1)))
	
def logisWithData(dataArr):
	weights = stocGradAscent0(dataArr,labelMat)
	testDataArr = loadTestDataSet(testFileName,separator)
	return uniform(sigmoid(sum(testDataArr*weights,axis=1)))
		
	
#dataArr,labelMat = loadDataSet()
#weights = stocGradAscent0(dataArr,labelMat)
#print(weights)
#plotBestFit(weights)

