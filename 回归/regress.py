'''
	本章 预测 数值型 结果，成为回归
	
'''
from numpy import *

#加载标准回归的数据
def loadDataSet(fileName):
	numFeat = len(open(fileName).readline().split('\t')) - 1
	dataMat = []
	labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat,labelMat
	
#标准回归函数 标准回归是线性回归
def standRegress(xArr,yArr):
	xMat = mat(xArr)
	yMat = mat(yArr).T
	xTx = xMat.T * xMat
	if linalg.det(xTx) == 0.0:
		print("this matrix is singular, cannot do inverse")
		return
	ws = xTx.I * (xMat.T*yMat)
	return ws
	
xArr,yArr = loadDataSet('ex0.txt')
ws = standRegress(xArr,yArr)
print(ws)

#############################


	