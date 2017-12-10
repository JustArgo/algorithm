'''	

	递归神经网络-narx
	
'''
class narx:
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
	
	def train(self):
	
	def train_parrel(self,X):
		hidden_output = []
		hidden_hidden_output = []
		output = []
		for num in range(self.epochs):
			for t in range(self.base_length):
				if t==0:
					hidden_output[:,t] = self.weights_input_hidden.T * X[:,t]
				else:
					hidden_hidden_output[:,t] = self.weights_input_hidden.T * X[:,t] + self.weights_hidden_hidden.T * hidden_output[:,t-1] 
				
				for n in range(self.layers[1]):
					hidden_hidden_output[n,t] = 1.0/(1+exp(-hidden_output(n,:)))
		
				output[:,t] = self.weights_hidden_hidden.T * hidden_hidden_output[:,t]
				
				error = output[:,t]-y[:,t];
				self.error_cost[0,num] = sum(error**2)
				if(self.error_cost[0,num]<self.epsr):
					break
				self.update_weights(error,hidden_hidden_output,X,t)
				
	def update_weights(self,error,hidden_hidden_output,X,t):
		
		