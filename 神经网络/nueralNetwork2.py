'''
	神经网络
'''
import numpy as np
from math import *
def tanh(x):
	return np.tanh(x)
	
def tanh_deriv(x):
	return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x):
	return 1/(1+(np.exp(-x)))
	
def logistic_deriv(x):
	return logistic(x)*(1-logistic(x))

class neuralNetwork:
	def __init__(self,layers,activation='tanh'):
		
		if activation == 'tanh':
			self.activation = tanh
			self.activation_deriv = tanh_deriv
		if activation == 'logistic':
			self.activation = logistic
			self.activation_deriv = logistic_deriv
		
		self.layers = layers
		
		self.weights = []
		
		for i in range(1,len(layers)-1):
			self.weights.append((2*np.random.random((layers[i-1]+1,layers[i]+1))-1)*0.25)
			self.weights.append((2*np.random.random((layers[i]+1,layers[i+1]))-1)*0.25)
	
	#参数说明  X是m行n列  y是m行1列
	def fit(self,X,y,learning_rate=0.002,epochs=100):
		
		temp = np.ones((X.shape[0],X.shape[1]+1))
		temp[:,0:-1] = X
		X = temp	#temp的最后一列都是1，代表bias
		y = np.array(y)
		for i in range(epochs):
			index = np.random.randint(X.shape[0])
			a = [X[index]]
			for j in range(len(self.weights)):#有两个权重就计算两次
				#print(self.activation(np.dot(a[-1],self.weights[j])))
				a.append(self.activation(np.dot(a[j],self.weights[j])))
			
			#计算偏差
			# deltas 代表的是  round(E)/round(Net)
			err = y[index] - a[-1]
			deltas =  [err * self.activation_deriv(a[-1])]  
			
			#计算隐藏层到输入层的deltas,进行反向传播权重
			for j in range(len(self.layers)-2,0,-1):
				deltas.append(deltas[-1].dot(self.weights[j].T) * self.activation_deriv(a[j]))  #计算要分清楚 数值类计算和矩阵类计算
			
			deltas.reverse() #翻转偏导数
			
			#重新计算weights
			for j in range(len(self.weights)):
				layer = np.atleast_2d(a[j])  
				delta = np.atleast_2d(deltas[j])
				self.weights[j] += learning_rate* layer.T.dot(delta)			#计算要分清楚 数值类计算和矩阵类计算
				
	def predict(self, x):         
		x = np.array(x)         
		temp = np.ones(x.shape[0]+1)         
		temp[0:-1] = x         
		a = temp         
		for l in range(0, len(self.weights)):             
			a = self.activation(np.dot(a, self.weights[l]))         
		return a		
	#大致 这个逻辑，具体为什么不懂，是因为没有清楚的数学逻辑思维
	'''

		1 假设有1个案例，输入层2层，隐藏层3层，输出层1层
		2 可以确定的是self.weights
			[
				[ 0.21 , 0.13, 0.15 
				  0.11 , 0.15, 0.17	
				  0.17 , 0.25, 0.12 ],    #3行3列
				  
				[ 0.21 
				  0.11 
				  0.18 
				  0.23 ]			  		#4行1列
			]
		
		3 更新
			10000次迭代  ，假设有100条输入数据
			
				a = 100 * 3
				
				[
					[100行*3列]
				
				]
				
				for j self.weights:
					a.append (a * self.weights[j])
				
					第一次迭代  * 3行4列
						
						[
							[100行*3列],
							[100行*4列],
						]
						
					第二次迭代  * 4行1列
						
						[
							[100行*3列],
							[100行*4列],
							[100行*1列]
						]
					
					得到100个偏差 y估 - y 
				#计算deltas	
					
					Err = y估 - y
					[
					  0.7
					 -0.2     100行1列
					 ...
					]
					delta() = Err * round(J)/round(y)   激活函数对激活之前的输出进行求导
					也是 100行1列
					self.weights += learning_rate * np.dot(delta.T * x)
					
					
					
					
					
	'''		

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

#矩阵类计算用 matrix
#数值类计算用 array	
def train():
	dataMat,labelMat = loadTrainDataSet()	
	nn = neuralNetwork([2,3,1])
	nn.fit(np.mat(dataMat),np.mat(labelMat).T)
	#print(nn.weights)
	print(nn.predict([4,1]))
	print(nn.predict([2,2]))
	print(nn.predict([3,3]))
	print(nn.predict([1,4]))
	print(nn.predict([5,3]))
	print(nn.predict([2,3]))
		