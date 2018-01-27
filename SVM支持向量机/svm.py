'''
	svm支持向量机 求分类问题 二值分类问题
	1 以二维点 说明  求超平面 即 直线 f(x) = W(T) * x + b 使得距离该直线最近的点 到 该直线的间隔最大
	2  距离为 z(x) = w(T) * x + b / ||w||
	3 转化为 求  1/||w||的最大值 
	4 转化为 求  ||w||的最小值  相当于 求 1/2 ||w||平方 的最小值  s.t., y(i)*(w(T) * x + b) >= 1,i=1,2,3...n
	5 引入拉格朗日 乘子 L(w,b,alpha) = 1/2 ||w||平方 - sigma(1,n) alpha(i)(y(i)(w(T) * x(i) + b)-1)
	6 对 w 求导 == 0  得到  w = sigma(1,n) alpha(i)y(i)x(i)
	7 对 b 求导 == 0 得到   sigma(1,n) alpha(i)y(i) = 0
	8 代入 拉格朗日表达式 并 转化成对偶问题 sigma(1,n) alpha(i) - 1/2 sigma(i,j)(1,n) alpha(i)alpha(j)y(i)y(j)x(i)(T)x(j)
	9 求 上式的极大 转为 求alpha的极大
	10 为什么 i 和 j能对偶
	
	*
		C的含义是什么
		
		
	
'''
from numpy import *
import plot
#加载数据
def loadDataSet(fileName):
	dataMat = []
	labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]),float(lineArr[1])])
		labelMat.append(float(lineArr[2]))
	return dataMat,labelMat


#启发式 选择 j
def selectJrand(i,m):
	j = i
	while (j==i):
		j = int(random.uniform(0,m))
	return j

#修正alpha 值
def clipAlpha(aj,H,L):
	if aj > H:
		aj = H
	if aj < L:
		aj = L
	return aj

#简化版 smo算法
# C代表 alpha的上限
# toler 代表容错率
# maxIter 代表最大迭代次数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
	dataMatrix = mat(dataMatIn)
	labelMat = mat(classLabels).transpose()
	b = 0
	m,n = shape(dataMatrix)
	alphas = mat(zeros((m,1)))
	#exit()
	iter = 0
	while (iter<maxIter):
		alphaPairsChanged = 0
		
		for i in range(m):
			#fXi代表估算值     
			fXi = float(multiply(alphas,labelMat).T*dataMatrix * dataMatrix[i,:].T) + b
			#Ei 代表错误差值
			Ei = fXi - float(labelMat[i])
			#差值必须在一定程度以上 才有必要进行下一轮迭代
			if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i]>0)):
				j = selectJrand(i,m)
				fXj = float(multiply(alphas,labelMat).T * (dataMatrix * dataMatrix[j,:].T)) + b
				Ej = fXj - float(labelMat[j])
				alphaIold = alphas[i].copy()
				alphaJold = alphas[j].copy()
				if (labelMat[i] != labelMat[j]):
					L = max(0,alphas[j] - alphas[i])
					H = min(C, C + alphas[j] - alphas[i])
				else:
					L = max(0,alphas[j] + alphas[i] - C)
					H = min(C, alphas[j] + alphas[i])
				if L==H:
					#print("L==H")
					continue
				#为什么eta 必须 < 0
				eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - dataMatrix[i,:] * dataMatrix[i,:].T - dataMatrix[j,:] * dataMatrix[j,:].T
				if eta >= 0:
					#print("eta>=0")
					continue
				alphas[j] -= labelMat[j]*(Ei-Ej)/eta
				alphas[j] = clipAlpha(alphas[j],H,L)
				if (abs(alphas[j]-alphaJold) < 0.00001):
					#print("j not moving enough")
					continue
				alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
				b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold) * dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
				b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold) * dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
				if (0 < alphas[i] and C > alphas[i]):
					b = b1
				elif (0 < alphas[j] and C > alphas[j]):
					b = b2
				else:
					b = (b1 + b2)/2.0
				alphaPairsChanged += 1
				#print("%d, %d, %d" % (iter,i,alphaPairsChanged))
		#iter += 1
		if (alphaPairsChanged == 0):
			iter += 1
		else:
			iter = 0
		#print("iteration number: %d" % iter)
	return b,alphas

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
		labelMat.append(float(lineList.pop()))
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

#计算ws 超平面	
def calcWs(alphas,dataArr,classLabels):
	X = mat(dataArr)
	labelMat = mat(classLabels).transpose()
	m,n = shape(X)
	w = zeros((n,1))
	for i in range(m):
		w += multiply(alphas[i]*labelMat[i],X[i,:].T)
	return w

#计算分类值
def classify(testDataArr,w,b):
	valList = []
	for val in testDataArr:
		valList.append(val * mat(w)+b)
	return valList
	

#svm算法
def svm(trainFileName='trainSet.txt',testFileName='testSet.txt',separator='\t'):
	dataArr,labelArr = loadTrainDataSet(trainFileName,separator)
	
	#exit()
	#  alpha上限  0.6
	#  误差值上限 0.001
	#  最大迭代次数 40
	b,alphas = smoSimple(dataArr,labelArr,1.5,0.001,40)
	#for i in range(len(alphas)):
	#	if alphas[i] != 0:
	#		print('%d %f' % (i,alphas[i]))
	w = calcWs(alphas,dataArr,labelArr)
	#print(w,b)
	
	#for i in range(100):
	#	if alphas[i]>0.0:
	#		print(dataArr[i],labelArr[i])
	testDataArr = loadTestDataSet(testFileName)
	#w = calcWs(alphas,testDataArr,labelArr)
	w[0,0] += b
	plot.plotPointAndLine(mat(dataArr)[:,1:],w)
	#return classify(testDataArr,w,b),w,b
	
#dataArr,labelArr = loadDataSet("testSet.txt")
#print(shape(labelArr))
#b,alphas = smoSimple(dataArr,labelArr,0.6,0.001,40)

#for i in range(100):
#	if alphas[i]>0.0:
#		print(dataArr[i],labelArr[i])
		