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
	return mat(dataArr)import matplotlib.pyplot as plt
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
	
def plotPointAndLine(dataArr,w):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(dataArr.T.A[0],dataArr.T.A[1],s=30,c='black')
	x = arange(2,8,0.1)
	y = (-w[0]-w[1]*x)/w[2]
	ax.plot(x,y)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()
	
def plotRect(valArr,labelArr):
	width = 0.4
	length = len(valArr)
	ind = linspace(0.5,0.5+length-1,length)
	# make a square figure
	fig = plt.figure(1)
	ax  = fig.add_subplot(111)
	# Bar Plot
	ax.bar(ind,valArr,width,color='green')
	# Set the ticks on x-axis
	ax.set_xticks(ind)
	ax.set_xticklabels(labelArr)
	# labels
	ax.set_xlabel('feature')
	ax.set_ylabel('average price')
	# title
	ax.set_title('Relation', bbox={'facecolor':'0.8', 'pad':5})
	plt.grid(True)
	plt.show()
	#plt.savefig("bar.jpg")
	plt.close()
