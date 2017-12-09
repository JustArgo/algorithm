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
		self.error_cost = zeros((1,epochs))
		self.weights_input_hidden = np.random.random((layers[0],layers[1]))
		self.weights_hidden_hidden = np.random.random((layers[1],layers[1]))
		self.weights_hidden_output = np.random.random((layers[1],layers[2]))
		
	def train(self,X,y):
		hidden_out = 0
		hidden_hidden_out = 0
		out = 0
		
		for num in range(self.epochs):
			for t in range(self.base_length):
				if t==0:
					hidden_out = self.weights_input_hidden.T * X[t]
				else:
					hidden_hidden_out = self.weights_input_hidden * X[t] + self.weights_hidden_hidden.T * hidden_hidden_out[t-1]
				
				for n in range(self.layers[1]):
					hidden_hidden_out[n,t] = 1.0/(1+exp(-hidden_out(n,:)))
					
				out[:,t] = self.weights_hidden_output.T * hidden_hidden_out[:,t]
				error = output[:,t]-y[:,t];
				self.error_cost[1,num] = sum((out[:,t]-y[:,t])**2)
				if(self.error_cose[1,num]<self.epsr):
					break
				self.update_weights(error,hidden_hidden_out,X,t)
					
			if(self.error_cost[1,num]<self.epsr):
				break
			
			
	def update_weights(self,error,hidden_hidden_out,X,t):
		hidden_output_temp = self.weights_hidden_output
		hidden_input_temp  = self.weights_input_hidden
		hidden_hidden_temp = self.weights_hidden_hidden
		for n in range(len(error)):
			delta_weights_ho[n,:] = 2*error[n,0].dot(hidden_hidden_out[:,0].T)
		hidden_ouput_temp = hidden_output_temp - self.eta * delta_weights_ho
		for n in range(length(error)):
			for m in range(len(hidden_hidden_out)):
				delta_weights_ih[:,m] = 2*error[n,0].dot(hidden_output_temp[n,m])*X[:,0]
			hidden_input_temp = hidden_input_temp - eta * delta_weights_ih.T	
		
		if(t!=1):
			for n in range(len(error)):	
				for m in range(len(hidden_hidden_out)):
					delta_weights_hh[m,:] = 2*error[n,0].dot(self.weights_hidden_ouput[n,m])*hidden_hidden_out[:,t-1].T
				hidden_hidden_temp = hidden_hidden_temp - self.eta * delta_weights_hh 
		
		self.weights_hidden_output = hidden_output_temp
		self.weights_input_hidden  = hidden_input_temp
		self.weights_hidden_hidden = hidden_hidden_temp
		
	def predict(self,x,y):
		