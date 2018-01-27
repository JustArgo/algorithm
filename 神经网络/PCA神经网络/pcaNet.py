'''

	pca神经网络
	
	1 传进来 样本
	2 进行网络训练
	3 得到第一主成分分量
	
'''
import numpy as np
#零均值化
def zeroMean(dataMat):      
    meanVal=np.mean(dataMat,axis=0)     #按列求均值，即求各个特征的均值
    newData=dataMat-meanVal
    return newData
	
def update(x,w,y,eta):
	#print(y[0]*w)
	x = np.atleast_2d(x)
	#print(((x.T - y[0] * w)).shape)
	w = w + eta * y[0] * (x.T - y[0] * w)
	#w = y.dot(w)
	#print(x)
	return w
class pcaNet:
	def __init__(self,X,eta=0.01,epsr=0.2,epoch=2):
		
		self.X = X
		
		self.eta = eta
		self.epsr = epsr
		self.epoch = epoch
		self.weights = np.random.random((3,1))
		self.length = X.shape[0]
		
	def fit(self):
		for k in range(self.epoch):
			for i in range(self.length):
				temp = self.weights 
				#if k==0 and i==0:
					#print(self.weights.T.dot(self.X[i].T))
				self.weights = update(self.X[i],self.weights,self.weights.T.dot(self.X[i]),self.eta)
				#print(self.weights-temp)
				if np.linalg.norm(self.weights-temp)<self.epsr:
					print(k,i)
					break;
	
def loadTrainDataSet(fileName='train.txt'):
	dataMat = []
	labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split("\t")
		fltLine = []
		for i in range(len(curLine)-1):
			fltLine.append(float(curLine[i]))
		
		dataMat.append(fltLine)
		labelMat.append(float(curLine[-1]))
	return dataMat,labelMat	
	
def train():
	dataMat,labelMat = loadTrainDataSet()
	pca = pcaNet(np.array(zeroMean(dataMat)))
	pca.fit();
	print(pca.weights.T*dataMat[0])
	
	
train()
	