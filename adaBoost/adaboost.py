'''
adaboost元算法(二值分类器)
选取之前的某个算法为 弱分类器  此处选取 单层决策树
1 在n个特征中循环，判断哪个特征 在['lt','gt'] threshValue 条件下 判断更好
2 把第1步返回的 结果 当成预测结果，进行错误率的计算  循环n个特征之后，  得到特定的 列特征，lt或gt , threshValue    minErrorRate
3 用第二步得到的 minErrorRate 为当前弱分类器计算  alpha 即权重  alpha是基础权重
4 先计算下一次迭代的D  m行1列  代表数据的m个向量的权重，累加计算错误率，如果错误率 没有达到0.0  则继续算下一个弱分类器
5 返回最后计算的 x个 弱分类器
6 假设现在又一个样本 要进行分类，则传入样本，和第5步中的x个分类器，累加计算分类结果值，>0 则为 1类, <0 则为-1类
'''
from numpy import *
import plot
#加载 自定义数据
def loadSimpleData():
	dataMat = matrix([
	[1,  2.1],
	[2,  1.1],
	[1.3, 1],
	[1,  1],
	[2,  1],
	[3,  1]
	])
	classLabels = [1.0,1.0,-1.0,-1.0,1.0,1.0]
	return dataMat,classLabels
	
#该函数用于 计算特定 列特征，阈值，比较形式下的 分类结果
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
	retArray = ones((shape(dataMatrix)[0],1))
	if threshIneq == 'lt':
		#print(dataMatrix[:,dimen])
		#print(type(dataMatrix[:,dimen]))
		retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:,dimen] > threshVal] = -1.0
	return retArray
	
#构建弱分类器，计算最小错误率，该分类器下的预测结果
def buildStump(dataArr,classLabels,D):
	dataMatrix = mat(dataArr)
	labelMat = mat(classLabels).T
	m,n = shape(dataMatrix)
	numSteps = 10.0
	bestStump = {}
	bestClassEst = mat(zeros((m,1)))
	minError = inf
	for i in range(n):
		rangeMin = dataMatrix[:,i].min()
		rangeMax = dataMatrix[:,i].max()
		stepSize = (rangeMax-rangeMin)/numSteps
		for j in range(-1,int(numSteps)+1):
			for inequal in ['lt','gt']:
				threshVal = (rangeMin + float(j) * stepSize)
				predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
				errArr = mat(ones((m,1)))
				errArr[predictedVals == labelMat] = 0
				weightedError = D.T * errArr
				#print("split: dim %d, thresh %2.f, thresh inequal: %s, the weighted errors is %.3f" % (i,threshVal,inequal,weightedError))
				if weightedError < minError:
					minError = weightedError
					bestClasEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	return bestStump,minError,bestClasEst
	
#基于单层决策树的AdaBoost训练过程
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
	weakClassArr = []
	m = shape(dataArr)[0]
	D = mat(ones((m,1))/m)
	aggClassEst = mat(zeros((m,1)))
	for i in range(numIt):
		#print(D)
		bestStump,error,classEst = buildStump(dataArr,classLabels,D)
		#print("D:",D.T)
		#print(error)
		alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
		#print(alpha)
		bestStump['alpha'] = alpha
		weakClassArr.append(bestStump)
		#print("classEst: ",classEst.T)
		expon = multiply(-1*alpha*mat(classLabels).T,classEst)
		D = multiply(D,exp(expon))
		D = D/D.sum()
		aggClassEst += alpha*classEst
		#print("aggClassEst:",aggClassEst.T)
		aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
		errorRate = aggErrors.sum()/m
		#print("total error: ",errorRate,"\n")
		if errorRate == 0.0:
			#print("i:",i)
			break
	return weakClassArr
	
#用训练出来的弱分类器数组 进行分类
def adaClassify(datToClass,classifierArr):
	dataMatrix = mat(datToClass)
	m = shape(dataMatrix)[0]
	aggClassEst = mat(zeros((m,1)))
	for i in range(len(classifierArr)):
		classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
		aggClassEst += classifierArr[i]['alpha']*classEst
		#print aggClassEst
	return sign(aggClassEst)


#加载训练数据
def loadTrainDataSet(trainFileName,separator='\t'):
	fopen = open(trainFileName)
	lineArr = []
	labelArr = []
	for line in fopen.readlines():
		lineList = line.strip().split(separator)
		featArr = []
		for i in range(len(lineList)-1):
			featArr.append(float(lineList[i]))
		lineArr.append(featArr)
		labelArr.append(float(lineList[len(lineList)-1]))
	return lineArr,labelArr

#加载测试数据
def loadTestDataSet(testFileName,separator='\t'):
	fopen = open(testFileName)
	lineArr = []
	for line in fopen.readlines():
		lineList = line.strip().split(separator)
		lineArr.append(list(map(float,lineList)))
	return lineArr	

	
#adaboost分类
def adaBoost(trainFileName='trainSet.txt',testFileName='testSet.txt',separator='\t'):
	dataArr,labelArr = loadTrainDataSet(trainFileName,separator)
	#plot.plotPoint(mat(dataArr))
	#print(dataArr)
	classifierArr = adaBoostTrainDS(dataArr,labelArr,30)
	testDataArr = loadTestDataSet(testFileName,separator)
	return adaClassify(testDataArr,classifierArr)

#开始进行算法的测试
#dataArr,labelArr = loadSimpleData()
#classifierArr = adaBoostTrainDS(dataArr,labelArr,30)
#classifyRet = adaClassify([[5,0],[0,0]],classifierArr)
#print(classifyRet)