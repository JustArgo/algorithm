'''
循环神经网络:ELMAN

'''
import numpy as np

class elman:
	#参数说明
	# base_length 代表要用几个样本来 预测下一个样本
	def __init__(self,layers,eta=0.1,epochs=4000,epsr=0.0001,base_length=3):
		self.layers = layers
		self.eta = eta
		self.epochs = epochs
		self.epsr = epsr
		self.base_length = base_length
		self.error_cost = np.zeros((1,epochs))
		self.weights_input_hidden = np.random.random((layers[0],layers[1]))
		self.weights_hidden_hidden = np.random.random((layers[1],layers[1]))
		self.weights_hidden_output = np.random.random((layers[1],layers[2]))
		
	def train(self,X,y):
		hidden_out = np.zeros((self.layers[0],self.base_length))
		hidden_hidden_out = np.zeros((self.layers[1],self.base_length))
		output = np.zeros((self.layers[2],self.base_length))
		
		for num in range(self.epochs):
			for t in range(self.base_length):
				if t==0:
					hidden_out = self.weights_input_hidden.T * X[:,t]
				else:
					hidden_hidden_out[:,t] = np.atleast_2d(X[:,t]).dot(self.weights_input_hidden)  + self.weights_hidden_hidden.dot(hidden_hidden_out[:,t-1])
					#print(hidden_hidden_out)
				
				for n in range(self.layers[1]):
					#print(1.0/(1+np.exp(-1*hidden_out[n,0])))
					hidden_hidden_out[n,t] = 1.0/(1+np.exp(-1*hidden_out[n,t]))
					
				#print(hidden_hidden_out[:,t].shape)
				#print(self.weights_hidden_output.T * hidden_hidden_out[:,t])
				#print((np.atleast_2d(hidden_hidden_out[:,t])).shape)
				output[:,t] = self.weights_hidden_output.T.dot(np.atleast_2d(hidden_hidden_out[:,t]).T)[0]
				error = output[:,t]-y[:,t];
				self.error_cost[0,num] = sum(error**2)
				if(self.error_cost[0,num]<self.epsr):
					break
				self.update_weights(error,hidden_hidden_out,X,t)
					
			if(self.error_cost[0,num]<self.epsr):
				break
			
			
	def update_weights(self,error,hidden_hidden_out,X,t):
		hidden_output_temp = self.weights_hidden_output
		hidden_input_temp  = self.weights_input_hidden
		hidden_hidden_temp = self.weights_hidden_hidden
		delta_weights_ho = np.zeros((self.layers[1],self.layers[2]))
		delta_weights_ih = np.zeros((self.layers[0],self.layers[1]))
		#print(error)
		for n in range(len(error)):
			#print((2 * error[n] * hidden_hidden_out[:,t].T).shape)
			# 2 * [18,1] * [1,4]
			delta_weights_ho[:,n] = 2 * hidden_hidden_out[:,t] * np.atleast_2d(error[n])
		hidden_output_temp = hidden_output_temp - self.eta * delta_weights_ho
		delta_weights_ih = 2 * np.atleast_2d(X[:,t]).T.dot(np.atleast_2d(error)).dot(hidden_output_temp.T)
		hidden_input_temp = hidden_input_temp - self.eta * delta_weights_ih	
		
		if(t!=0):
			delta_weights_hh = (2 * np.atleast_2d(error).dot(self.weights_hidden_output.T)).T * hidden_hidden_out[:,t-1]
			hidden_hidden_temp = hidden_hidden_temp - self.eta * delta_weights_hh 
		
		self.weights_hidden_output = hidden_output_temp
		self.weights_input_hidden  = hidden_input_temp
		self.weights_hidden_hidden = hidden_hidden_temp
		
	def predict(self,X,y):
		print("predict")
		
def loadTrainDataSet(fileName='train.txt'):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split("\t")
		fltLine = []
		for i in range(len(curLine)):
			fltLine.append(float(curLine[i]))
		dataMat.append(fltLine)
	return dataMat	

def dealData(dataMat):
	dataArr = np.atleast_2d(dataMat)
	m,n = dataArr.shape
	transArr = np.zeros(((m-3)*4,3))
	realOutput = np.zeros((4,3))
	for j in range(3):
		concat = np.concatenate([dataArr[j],dataArr[j+1],dataArr[j+2]])
		#print(transArr[:,j].shape)
		transArr[:,j] = np.atleast_2d(concat)[0]
		#print(np.atleast_2d(concat)[0])
		realOutput[:,j] = np.atleast_2d(dataArr[j+3]).T.tolist()[0]
	return transArr,realOutput
		
def main():
	dataMat = loadTrainDataSet()
	dataMat,y = dealData(dataMat)
	elmanNet = elman([12,18,4])
	elmanNet.train(dataMat,y)
	
main()