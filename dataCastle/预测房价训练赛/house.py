from numpy import *
import treeRegress as tr
import regress as rg
import json
import plot

#加载所有的数据
def loadDataSet(fileName):
	fopen = open(fileName)
	dataArr = []
	for line in fopen.readlines():
		dataArr.append(line.strip().split(','))
	fopen.close()
	return dataArr


#加载并预处理数据集
def loadTrainDataSet(trainFileName):
	fopen = open(trainFileName,'r')
	dataArr = []
	labelArr = []
	for line in fopen.readlines():
		lineList = line.strip().split()
		tmpList = list(map(float,lineList[2:12]))
		tmpList.append(float(lineList[1]))
		dataArr.append(lineList)
	return dataArr

#加载并预处理数据集
def loadTestDataSet(testFileName):
	fopen = open(testFileName)
	dataArr = []
	labelArr = []
	for line in fopen.readlines():
		lineList = line.strip().split(',')
		dataArr.append(list(map(float,lineList[1:12])))
	fopen.close()
	return dataArr	

#将模型保存到文件中
def saveModel2File(myTree,fileName):
	output = open(fileName,'w')
	output.write(json.dumps(myTree))
	output.close()
	
#加载模型
def loadModel():
	fopen = open('model.txt')
	line = fopen.readline()
	if len(line)>0:
		return json.loads(line)
	else:
		return None

#处理时间 传矩阵进来
def filterDate(dataArr):
	length = len(dataArr)
	for i in range(length):
		if int(dataArr[i,10]) > 0:
			dataArr[i,10] = 1
	return dataArr

#删除作用不大的列	
def delUnusefulFeat(dataArr):
	dataArr = delete(dataArr,[2,3],axis=1)
	return dataArr

#调整列顺序
def adjustData(dataArr):
	sellPrice = dataArr[:,1]
	dataArr = dataArr[:,2:11]
	dataArr = column_stack((dataArr,sellPrice))
	return dataArr
	
#数据转换成float
def tran2Float(dataArr):
	m,n = shape(dataArr)
	dataArr = dataArr.A.tolist()
	for i in range(m):
		for j in range(n):
			dataArr[i][j] = float(dataArr[i][j])
	
	return mat(dataArr)
		
#过滤数据
def filterData(dataArr):
	dataArr = filterDate(dataArr)
	dataArr = delUnusefulFeat(dataArr)
	dataArr = adjustData(dataArr)
	dataArr = tran2Float(dataArr)
	#for i in range(shape(dataArr)[0]):
	#	dataArr[i,-1] = dataArr[i,-1]/10000
	return dataArr
		
#过滤测试数据
def filterTestData(dataArr):
	length = len(dataArr)
	for i in range(length):
		if int(dataArr[i,9]) > 0:
			dataArr[i,9] = 1
		
	dataArr = delete(dataArr,[0,1],axis=1)
	dataArr = tran2Float(dataArr)
	return dataArr		
		
#对数据进行分析
def analyze():
	# 不同月份的 每平方多少钱
	dataMat = mat(loadTrainDataSet('bak.txt'))
	m,n = shape(dataMat)
	'''
	for i in range(m):
		dataMat[i,0] = int(str(dataMat[i,0])[4:6])
		print(str(dataMat[i,0]))
	
	month = range(1,13)
	average = []
	for i in range(1,13):
		square = 0
		price  = 0
		for j in range(len(dataMat)):
			if int(dataMat[j,0]) == i:
				price += int(dataMat[j,1])
				square += int(dataMat[j,4])
		average.append(price/square)
	#plot.plotRect(average,month)
	
	#按经纬度划分
	for i in range(m):
		dataMat[i,-2] = float(dataMat[i,-2])
	maxLat = float(max(dataMat[:,-2])[0,0])
	minLat = float(min(dataMat[:,-2])[0,0])
	print(maxLat,minLat)
	step = (maxLat - minLat) / 10 
	latlotAverage = []
	latlotLabel = []
	for i in range(10):
		square = 0
		price = 0
		for j in range(len(dataMat)):
			if float(dataMat[j,-2]) >= minLat + i * step and float(dataMat[j,-2]) < minLat + (i+1) * step:
				price += int(dataMat[j,1])
				square += int(dataMat[j,4])
		latlotAverage.append(price/square)
		latlotLabel.append(minLat + i * step)
	plot.plotRect(latlotAverage,latlotLabel)
	'''
	
	#按纬度划分
	'''
	for i in range(m):
		dataMat[i,-1] = float(dataMat[i,-1])
	maxLat = float(min(dataMat[:,-1])[0,0])
	minLat = float(max(dataMat[:,-1])[0,0])
	print(maxLat,minLat)
	step = (maxLat - minLat) / 10 
	print(step)
	latlotAverage = []
	latlotLabel = []
	for i in range(10):
		square = 0
		price = 0
		for j in range(len(dataMat)):
			if float(dataMat[j,-1]) >= minLat + i * step and float(dataMat[j,-1]) < minLat + (i+1) * step:
				price += int(dataMat[j,1])
				square += int(dataMat[j,4])
		latlotAverage.append(0 if square == 0 else price/square)
		latlotLabel.append(minLat + i * step)
	plot.plotRect(latlotAverage,latlotLabel)
	'''
	
	#按分数划分
	'''
	maxScore = 0 #float(max(dataMat[:,7])[0,0])
	minScore = 9999 #float(min(dataMat[:,7])[0,0])
	for i in range(m):
		score = float(dataMat[i,7])
		if score > maxScore:
			maxScore = score
		if score < minScore:
			minScore = score
	print(maxScore,minScore)
	step = (maxScore-minScore)/10.0
	latlotAverage = []
	latlotLabel = []
	for i in range(10):
		square = 0
		price = 0
		for j in range(len(dataMat)):
			if float(dataMat[j,7]) >= minScore + i * step and float(dataMat[j,7]) < maxScore + (i+1) * step:
				price += int(dataMat[j,1])
				square += int(dataMat[j,8])
		latlotAverage.append(price/square)
		latlotLabel.append(str(minScore + i * step))
	plot.plotRect(latlotAverage,latlotLabel)
	'''
	
	#按楼层数划分
	'''
	maxScore = 0 #float(max(dataMat[:,7])[0,0])
	minScore = 9999 #float(min(dataMat[:,7])[0,0])
	for i in range(m):
		score = float(dataMat[i,6])
		if score > maxScore:
			maxScore = score
		if score < minScore:
			minScore = score
	print(maxScore,minScore)
	step = (maxScore-minScore)/10.0
	latlotAverage = []
	latlotLabel = []
	for i in range(10):
		square = 0
		price = 0
		for j in range(len(dataMat)):
			if float(dataMat[j,6]) >= minScore + i * step and float(dataMat[j,6]) < maxScore + (i+1) * step:
				price += int(dataMat[j,1])
				square += int(dataMat[j,8])
		latlotAverage.append(price/square)
		latlotLabel.append(str(minScore + i * step))
	plot.plotRect(latlotAverage,latlotLabel)
	'''
	
	#按建筑年份划分
	'''
	maxScore = 0 #float(max(dataMat[:,7])[0,0])
	minScore = 9999 #float(min(dataMat[:,7])[0,0])
	for i in range(m):
		score = float(dataMat[i,10])
		if score > maxScore:
			maxScore = score
		if score < minScore:
			minScore = score
	print(maxScore,minScore)
	step = (maxScore-minScore)/10.0
	latlotAverage = []
	latlotLabel = []
	for i in range(10):
		square = 0
		price = 0
		for j in range(len(dataMat)):
			if float(dataMat[j,10]) >= minScore + i * step and float(dataMat[j,10]) < maxScore + (i+1) * step:
				price += int(dataMat[j,1])
				square += int(dataMat[j,8])
		latlotAverage.append(price/square)
		latlotLabel.append(str(minScore + i * step))
	plot.plotRect(latlotAverage,latlotLabel)
	'''
	
	#按修复年份  有修复 和 无修复
	'''
	latlotAverage = []
	latlotLabel = []
	for i in range(2):
		square = 0
		price = 0
		for j in range(len(dataMat)):
			if float(dataMat[j,11]) > 0 and i==0:
				price += int(dataMat[j,1])
				square += int(dataMat[j,8])
			else:
				price += int(dataMat[j,1])
				square += int(dataMat[j,8])
		latlotAverage.append(price/square)
	latlotLabel.append("repari")
	latlotLabel.append("un repair")
	plot.plotRect(latlotAverage,latlotLabel)
	'''
	
	#按卧室数目划分
	'''
	column = 2
	maxNum = int(max(dataMat[:,column])[0,0])
	minNum = int(min(dataMat[:,column])[0,0])
	step = 1
	latlotAverage = []
	latlotLabel = []
	for i in range(maxNum):
		square = 0
		price = 0
		for j in range(len(dataMat)):
			if float(dataMat[j,column]) >= minNum + i * step and float(dataMat[j,column]) < maxNum + (i+1) * step:
				price += int(dataMat[j,1])
				square += int(dataMat[j,8])
		latlotAverage.append(price/square)
		latlotLabel.append(str(minNum + i * step))
	plot.plotRect(latlotAverage,latlotLabel)
	'''
	
	
	
		
#预测价格
def predict():
	
	myTree = loadModel()
	if myTree == None:
		trainMat = mat(loadTrainDataSet('bak.txt'))
		trainMat = filterData(trainMat)
		#myTree = tr.createTree(trainMat, tr.modelLeaf, tr. modelErr, ops=(1,200))
		myTree = tr.createTree(trainMat, ops=(1,100))
		saveModel2File(myTree,'model.txt')
	
	testMat = mat(loadTestDataSet('kc_test.csv'))
	testMat = filterTestData(testMat)
	#print(myTree)
	#yHat = rg.regress(trainMat,testMat)
	yHat = tr.createForeCast(myTree,testMat)
	#yHat = list(map(int,yHat * 10000))
	writeToFile('sample.csv',yHat)

#后处理
def postDeal():
	fopen = open('sample.csv')
	output = open('sample2.csv','w')
	for line in fopen.readlines():
		output.write(str(int(float(line)*10))+"\n")
		
	fopen.close()
	output.close()
	
def writeToFile(fileName,yHat):
	output = open(fileName, 'w')
	for i in range(len(yHat)):
		output.write(str(yHat[i])+'\n')
	output.close()
		
	
