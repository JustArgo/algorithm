'''
	k均值聚类的步骤
	在所有点中 随机找k个点，作为中心点，然后计算距离，调整中心点
	
'''
from numpy import *

#加载数据
def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split("\t")
		fltLine = []
		for i in range(len(curLine)):
			fltLine.append(float(curLine[i]))
		dataMat.append(fltLine)
	return dataMat

#距离计算方法 欧式距离
def distEclud(vecA,vecB):
	return sqrt(sum(power(vecA-vecB,2)))
	
#随机选择簇质心
def randCent(dataSet, k):
	n = shape(dataSet)[1]
	centroids = mat(zeros((k,n)))
	for j in range(n):
		minJ = min(dataSet[:,j])
		rangeJ = float(max(dataSet[:,j]) - minJ)
		centroids[:,j] = minJ + rangeJ * random.rand(k,1)
	return centroids
	
#kMeans聚类函数
def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m,2)))
	centroids = createCent(dataSet,k)
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		for i in range(m):
			minDist = inf
			minIndex = -1
			for j in range(k):
				distJI = distMeas(centroids[j,:],dataSet[i,:])
				if distJI < minDist:#1
					minDist = distJI
					minIndex = j
			if clusterAssment[i,0] != minIndex:#第一轮迭代 改变了centroids 第二轮进来如果没有走#1 则centroids没变
				clusterChanged = True
			clusterAssment[i,:] = minIndex,minDist**2
		#print(centroids)
		for cent in range(k):
			ptsInClust = dataSet[nonzero(clusterAssment[:,0]==cent)[0]]
			centroids[cent,:] = mean(ptsInClust, axis=0)
	return centroids,clusterAssment
	
'''
	二分k均值聚类算法
	1 最开始只有一个簇，分成两个簇之后 sse会比一个簇更小
	2 再循环两个簇，每个簇进行划分之后，如果 sse(误差平方和) 大于 没有划分的sse则不划分
	3 循环第2步 直至有k个簇
'''
#二分K-均值
def biKMeans(dataSet,k,distMeas=distEclud):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m,2)))
	centroid0 = mean(dataSet,axis=0).tolist()[0]
	#print(type(centroid0))
	centList = [centroid0]
	for j in range(m):
		#print(j)
		#print(dataSet)
		aa = dataSet[0]
		#print(type(aa))
		clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
	while (len(centList)<k):
		lowestSSE = inf
		for i in range(len(centList)):
			ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
			centroidMat,splitClustAss = kMeans(ptsInCurrCluster,2,distMeas)
			sseSplit = sum(splitClustAss[:,1])
			sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
			if (sseSplit + sseNotSplit) < lowestSSE:
				bestCentToSplit = i
				bestNewCents = centroidMat
				bestClustAss = splitClustAss.copy()
				lowestSSE = sseSplit + sseNotSplit
		bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)		
		bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
		#print('the bestCentToSplit is:',bestCentToSplit)
		#print('the len of bestClustAss is: ',len(bestClustAss))
		centList[bestCentToSplit] = bestNewCents[0,:].tolist()
		centList.append(bestNewCents[1,:].tolist())
		#print(centList)
		clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
	return centList,clusterAssment

#测试算法
dataMat3 = loadDataSet('testSet2.txt')
#print(dataMat3)
centList,myNewAssments = biKMeans(mat(dataMat3),3)
print(centList)

 
			
	
	

	
	
	
	
	
	
	
	
	
	
	
	
	
	


	