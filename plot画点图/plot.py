import matplotlib.pyplot as plt
from numpy import *
def plotPoint(dataArr):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(dataArr.T.A[0],dataArr.T.A[1],s=30,c='black')
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()
	
#加载数据点
def loadDataArr(fileName='point.txt'):
	fopen = open(fileName)
	dataArr = []
	for line in fopen.readlines():
		dataArr.append(line.strip().split())
	return mat(dataArr)