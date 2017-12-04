'''
	
	意义：
	
	步骤：
		(一) 确定参数
			1 确定输入X
			2 确定输出Y,O
			3 初始化 输出层-隐含层 的连接权重W(kj)  
			4 初始化C(ji)  隐含层-输入层
			5 初始化D(j)
		(二) 计算隐含层神经元的输出 Z(j)
		(三) 计算输出层神经元的输出 Y[y1,y2,...yq]
		
		注意：最开始要给定 eta(学习因子),alpha,epsr(错误率阈值)
		
'''
import numpy as np

#高斯核函数
def gaussian(x):
	return np.exp(-(x*x))

#计算范数的值
def norms(x):
	return 1;
	
#求导数
def round(x):
	return 1;
	
class rbf:
	def __init__(self,X,Y,layers,eta=0.02,alpha=0.5,epsr=0.05,epochs=10000):
		
		self.p = layers[1]
		self.df = np.random.ranf()
		self.N = X.shape[0]
		self.layers = layers
		self.X = X
		self.Y = Y
		
		self.d = []
		self.c = []
		self.w = []
		self.z = []
		self.y = []
		
		self.dOld = []
		self.cOld = []
		self.wOld = []
		
		maxI = []
		minI = []
		maxK = []
		minK = []
		for i in range(layers[0]):
			maxI[i] = max(X[:,i])
			minI[i] = min(X[:,i])
		for k in range(layers[2]):
			maxK[k] = max(y[:,k])
			minK[k] = min(y[:,k])
		
		self.c = np.zeros((layers[1],layers[0]))
		self.d = np.zeros((layers[1],layers[0]))
		self.w = np.zeros((layers[2],layers[1]))
		
		self.cOld = np.zeros((layers[1],layers[0]))
		self.dOld = np.zeros((layers[1],layers[0]))
		self.wOld = np.zeros((layers[2],layers[1]))
		
		#初始化c,d
		for i in range(layers[0]):
			for j in range(layers[1]):
				self.c[j,i] = minI[i] + (maxI[i] - minI[i])/(2*self.p) + j * (maxI[i]-minI[i])/p
				sigma = 0
				for k in range(self.N):
					sigma += X[k,i]-self.c[j,i]
				self.d[j,i] = self.df * np.sqrt(sigma/self.N)
		#初始化w
		for j in range(layers[1]):
			for k in range(layers[2]):
				self.w[k,j] = minK[k] + (j+1) * (maxK[k]-minK[k])/(layers[2]+1)
			
	#先计算隐含层的输出 z，再计算输出层y
	def calc(self):
		for j in range(self.layers[1]):
			self.z.append(np.exp(-np.square(norms(self.X-self.c[j]/self.d[j]))))
		
		for k in range(self.layers[2]):
			sigma = 0
			for j in range(self.layers[1]):
				sigma += self.w[k,j] * self.z[j]
			self.y.append(sigma)
	
	def RMS(self):
		sigma = 0
		for i in range(self.N):
			for k in range(self.layers[2]):
				sigma += np.square(self.Y[k,i] - self.y[k,i])
		return np.sqrt(sigma/(self.layers[2]*self.N))		
				
	def update(self):
		
		for j in range(self.layers[1]):
			for k in range(self.layers[2]):
				tempW = self.w[k,j]
				self.w[k,j] = self.w[k,j] - self.eta * round(self.w[k,j]) + self.alpha * (self.w[k,j] - self.wOld[k,j])
				self.wOld[k,j] = tempW
	
		for i in range(self.layers[0]):
			for j in range(self.layers[1]):
				tempC = self.c[j,i]
				tempD = self.d[j,i]
				self.c[j,i] = self.c[j,i] - self.eta * round(self.c[j,i]) + self.alpha * (self.c[j,i] - self.cOld[j,i])
				self.d[j,i] = self.d[j,i] - self.eta * round(self.d[j,i]) + self.alpha * (self.d[j,i] - self.dOld[j,i])
				self.cOld[j,i] = tempC
				self.dOld[j,i] = tempD
				
	def fit(self,X,Y):
		
		for num in range(self.epochs):
			
			self.calc()
			
			rms  = self.RMS()
			if rms <= self.epsr:
				break;
			else:
				self.update()
		