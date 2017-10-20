'''
	树回归
	针对m行数据
		
		每一个列，取某个值，分成两半，大于给定值的在左边，小于给定值的在右边
		
		
		
'''

from numpy import *
#树节点类
class treeNode():
	def __init__(self,feat,val,right,left):
		featureToSplitOn = feat
		valueOfSplit = val
		rightBranch = right
		leftBranch = left

#加载数据		
def loadDataSet(fileName):
	dataMat = []
	fr = open(filename)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = map(float,curLine)
		dataMat.append(fltLine)
	return dataMat
	
#二元切分方法
def binSplitDataSet(dataSet,feature,value):
	mat0 = dataSet[nonzero(dataSet[:feature] > value)[0],:][0]
	mat1 = dataSet[nonzero(dataSet[:feature] <= value)[0],:][0]
	return mat0,mat1
	
#创建树的算法
def creatTree(dataSet,leatType=regLeaf,errType=regErr,ops=(1,4)):
	feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
	if feat == None:
		return val
	retTree = {}
	retTree['spInd'] = feat
	retTree['spVal'] = val
	lSet,rSet = binSplitDataSet(lSet,leafType,errType,ops)
	retTree['left'] = createTree(lSet,leafType,errType,ops)
	retTree['right'] = createTree(rSet,leafType,errType,ops)
	return retTree
	
#求 y值的均值
def regLeaf(dataSet):
	return mean(dataSet[:-1])
	
#求 y值的均方差 * m行
def regErr(dataSet):
	return var(dataSet[:,-1]) * shape(dataSet)[0]
	
#选择最好的切分方案 并返回切分方案 特征的索引和值
def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
	tolS = ops[0]
	tolN = ops[1]
	if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
		return None,leafType(dataSet)
	m,n = shape(dataSet)
	S = errType(dataSet)
	bestS = inf
	bestIndex = 0
	bestValue = 0
	for featIndex in range(n-1):
		for splitVal in set(dataSet[:,featIndex]):
			mat0,mat1 = binSplitDataSet(dataSet,featIndex,featVal)
			if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): #如果切分之后，某个子集个数少于规定的阈值 则不以当前featIndex和featVal切分
				continue
			newS = errType(mat0) + errType(mat1)
			if newS < bestS:
				bestIndex = featIndex
				bestValue = splitVal
				bestS = newS
	if (S - bestS) < tolS:
		return None,leafType(dataSet)
	mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
	if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): 
		return None,leafType(dataSet)
	return bestIndex,bestValue

#测试算法

myDat = loadDataSet('ex00.txt')
myMat = mat(myDat)
myTree = createTree(myMat)
print(myTree)

#测试多次切分
myDat1 = loadDataSet('ex0.txt')
myMat1 = mat(myDat1)
myTree1 = createTree(myMat1)
print(myTree1)


#树剪枝

#判断是否是一颗树
def isTree(obj):
	return type(obj).__name__=='dict')
	
#计算平均值
def getMean(tree):
	if isTree(tree['right']):
		tree['right'] = getMean(tree['right'])
	if isTree(tree['left']):
		tree['left'] = getMean(tree['left'])
	return tree['left']+tree['right']/2.0

#剪枝函数
def prune(tree,testData):
	if shape(testData)[0] == 0:
		return getMean(tree)
	if (isTree('right')) or (isTree(tree['left'])):
		lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
	if isTree(tree['left']):
		tree['left'] = prune(tree['left'],testData)
	if isTree(tree['right']):
		tree['right'] = prune(tree['right'],testData)
	if not isTree(tree['left']) and not isTree(tree['right']):
		lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])		#用测试集 根据当前树的 featIndex featVal 进行划分 计算差值平方和
		errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) + sum(power(rSet[:,-1] - tree['right'],2))
		treeMean = (tree['left']+tree['right'])/2.0
		errorMerge = sum(power(testData[:,-1]-treeMean,2))
		if errorMerge < errorNoMerge:
			print("merging")
			return treeMean
		else:
			return tree
	else:
		return tree
		
#测试剪枝效果
myDat2 = loadDataSet('ex2.txt')
myMat2 = mat(myDat2)
myTree2 = createTree(myMat2)
myDatTest = loadDataSet('ex2test.txt')
myMat2Test = mat(myDatTest)
prune(myTree2,myMat2Test)



'''
	模型树
'''
#计算 数据集的 转成 线性回归的 X,Y 和 回归系数
def linearSolve(dataSet):
	m,n = shape(dataSet)
	X = mat(ones((m,n)))
	Y = mat(ones((m,1)))
	X[:1:n] = dataSet[:,0:n-1]
	Y = dataSet[:,-1]
	xTx = x.T*x
	if linalg.det(xTx) == 0.0:
		raise NameError('this matrix is singular cannot do inverse,\n\
		try increase the second value of ops')
	ws = xTx.I * (X.T * Y)
	return ws,X,Y     #算出来的ws  是n行一列，n-1个特征 就有n行，包括第0行的 常数回归系数，类似于直线方程的截距

#获得 dataSet的线性回归系数	
def modelLeaf(dataSet):
	ws,X,Y = linearSolve(dataSet)
	return ws

#计算线性回归的 差值平方和	
def modelErr(dataSet):
	ws,X,Y = linearSolve(dataSet)
	yHat = X * ws
	return sum(power(Y-yHat,2))
	
#测试模型树
myMat3 = mat(loadDataSet('exp2.txt'))
myTree3 = createTree(myMat3,modelLeaf,modelErr,(1,10))
print(myTree3)


'''
	回归树 和 模型树的比较
	
'''
#回归树评估
def regTreeEval(model,inDat):
	return float(model)
	
#模型树评估
def modelTreeEval(model,inDat):
	n = shape(inDat)[1]
	X = mat(ones((1,n+1)))
	X[:,1:n+1]=inDat
	return float(X*model)

#用回归树 对数据进行计算 modelEval只是默认 regTreeEval也可以传其它 模型评估器 
def treeForeCast(tree, inData, modelEvel=regTreeEval):
	if not isTree(tree):
		return modelEval(tree,inData)
	if inData[tree['spInd']] > tree['spVal']:
		if isTree(tree['left']):
			return treeForeCast(tree['left'],inData,modelEval)
		else:
			return modelEval(tree['left'])
	else:
		if isTree(tree['right']):
			return treeForeCast(tree['right'],inData,modelEval)
		else:
			return modelEval(tree['right'])

#对testData进行计算			
def createForeCast(tree,testData,modelEval=regTreeEval):
	m = len(testData)
	yHat = mat(zeros((m,1)))
	for i in range(m):
		yHat[i,0] = treeForeCast(tree,mat(testData[i]),modelEval)
	return yHat
	
#对回归树 和 模型树进行比较
trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
myTree = createTree(trainMat, ops=(1,20))
yHat = createForeCast(myTree,testMat[:,0])  #这里采用默认的 回归树进行估计
co1 = corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
print('co1 is ',co1)

myTree2 = createTree(trainMat,modelLeaf,modelTreeError,(1,20))
yHat2 = createForeCast(myTree2,testMat[:,0],modelTreeEval)
co2 = corrcoef(yHat2,testMat[:,1],rowvar=0)[0,1]
print('co2 is ',co2)

#查看标准的线性回归效果如何
ws,X,Y = linearSolve(trainMat)
print('ws is ',ws)

for i in range(shape(testMat)[0]):
	yHat[i]=testMat[i,0]*ws[1,0] + ws[0,0]
	
co3 = corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
print('co3 is ',co3)







	

			
	
		

	

	


















