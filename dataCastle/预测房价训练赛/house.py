from numpy import *
import treeRegress as tr
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
	fopen = open(trainFileName)
	dataArr = []
	labelArr = []
	for line in fopen.readlines():
		lineList = line.strip().split(',')
		tmpList = list(map(float,lineList[2:12]))
		tmpList.append(float(lineList[1]))
		dataArr.append(tmpList)
	fopen.close()
	return dataArr

#加载并预处理数据集
def loadTestDataSet(testFileName):
	fopen = open(testFileName)
	dataArr = []
	labelArr = []
	for line in fopen.readlines():
		lineList = line.strip().split(',')
		dataArr.append(list(map(float,lineList[1:10])))
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

#对数据进行分析
def analyze():
	# 不同月份的 每平方多少钱
	dataMat = mat(loadDataSet('kc_train.csv'))
	
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
	
	
		
#预测价格
def predict():
	
	myTree = loadModel()
	if myTree == None:
		trainMat = mat(loadTrainDataSet('kc_train.csv'))
		myTree = tr.createTree(trainMat, ops=(1,100))
		saveModel2File(myTree,'model.txt')
	
	testMat = mat(loadTestDataSet('kc_test.csv'))
	yHat = tr.createForeCast(myTree,testMat[:,1:11])
	writeToFile('sample.csv',yHat)

def writeToFile(fileName,yHat):
	output = open(fileName, 'w')
	for i in range(len(yHat)):
		output.write(str(yHat[i])+'\n')
	output.close()
		
	
