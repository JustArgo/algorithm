'''
	
	1 有输入， 训练集有label
	2 输入是3维的，则神经网络的输入层为 3个神经元  
	3 mnist输出是 10个 神经元，识别 0 - 9 
	

'''



import numpy as np
import re 
 
#本方法针对那些不是 全部数值型的数据,并且有多个特征的 此处外加对dataCastle的特殊处理
def dealDataSet(trainFileName,separator):
	matcher = re.compile(r'^[-+]?[0-9]+\.?[0-9]*$')
	dataMat = []
	labelMat = []
	fr = open(trainFileName)
	for line in fr.readlines():
		lineArr = line.strip().split(separator)
		list = []
		#list.append(1.0)
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
 
#加载测试集数据
def loadTestSet(testFileName,separator):
	matcher = re.compile(r'^[-+]?[0-9]+\.?[0-9]*$')
	dataMat = []
	fr = open(testFileName)
	for line in fr.readlines():
		lineArr = line.strip().split(separator)
		list = []
		#list.append(1.0)
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
 
# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
 


def calcWeight():
	dataMat,labelMat = dealDataSet('pfm_train.csv',',')
	
	# input dataset
	X = np.mat(dataMat).A
	 
	# output dataset            
	y = np.mat(labelMat).A
	 
	# seed random numbers to make calculation
	# deterministic (just a good practice)
	np.random.seed(1)
	 
	# initialize weights randomly with mean 0
	syn0 = 2*np.random.random((30,1)) - 1
	#print(syn0)
	 
	for iter in xrange(5):
		# forward propagation
		l0 = X
		l1 = nonlin(np.dot(l0,syn0))
		#print(l1.shape)
		#exit()
		# how much did we miss?
		l1_error = y - l1

		# multiply how much we missed by the 
		# slope of the sigmoid at the values in l1
		#print(type(l1))
		lie = nonlin(l1,True)
		l1_delta = l1_error * lie

		# update weights
		syn0 += np.dot(l0.T,l1_delta)
	return syn0

#根据权重计算测试集合的数据	
def calcTest(testFileName,weight):
	testDataArr = loadTestSet(testFileName,',')
	return nonlin(np.dot(testDataArr,weight))
	
#将结果写到文件中
def writeToFile(fileName,outputData):
	output = open(fileName, 'w')
	for i in range(len(outputData)):
		output.write(str(int(outputData[i][0]))+'\n')
	output.close()
	
def calc():
	weight = calcWeight()
	testEstimate = calcTest('pfm_test.csv',weight)
	#print(type(testEstimate))
	writeToFile('sample_nn.txt',testEstimate)
	
	
	