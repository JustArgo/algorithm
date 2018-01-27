'''
	
	深度(信念)置信网络
	
'''
import numpy as np
import random
import sys
sys.path.append('../')
sys.path.append('../../')
import nueralNetwork2 as nn
import dataset.mnist as mt

def sigmrnd(x):
	return sigm(x) > np.random.randn(x.shape[0],x.shape[1])

def sigm(x):
	return 1/(1+np.exp(-x))

def rbmup(rbm, x):
    #sigm为sigmoid函数
    #通过隐层计算下一层
    return sigm(np.tile(rbm.c, (x.shape[0], 1)) + x.dot(rbm.w));
	
#训练rbm
class rbm:
	def __init__(self,w,vw,b,vb,c,vc,alpha,momentum):
		self.w = w
		self.vw = vw
		self.b = b
		self.vb = vb
		self.c = c
		self.vc = vc
		self.alpha = alpha
		self.momentum = momentum
		
	
class dbn:
	def __init__(self,layers,featNum,opts=(1,0,100,2)):
		n = featNum
		self.sizes = [n] + layers
		
		self.rbm = []
		
		for i in range(len(self.sizes)-1):
			w  = np.zeros((self.sizes[i],self.sizes[i+1]))
			vw = np.zeros((self.sizes[i],self.sizes[i+1]))	
			
			b = np.zeros((1,self.sizes[i]))
			vb = np.zeros((1,self.sizes[i]))
			
			c = np.zeros((1,self.sizes[i+1]))
			vc = np.zeros((1,self.sizes[i+1]))
			
			self.rbm.append(rbm(w,vw,b,vb,c,vc,opts[0],opts[1]))
	
	def rbmtrain(self,rbm,X,opts=(1,0,100,2)):
		batchsize = opts[2]
		epochs = opts[3]
		
		m = X.shape[0]
		numbatches = int(m/batchsize)
		kk = []
		for i in range(m):
			kk.append(i)
		random.shuffle(kk)
		for i in range(epochs):
			err = 0
			for j in range(numbatches):
				batch = X[kk[j*batchsize:(j+1)*batchsize],:]
				v1 = batch
				h1 = sigmrnd(np.tile(rbm.c,(batchsize,1))+v1.dot(rbm.w))
				v2 = sigmrnd(np.tile(rbm.b,(batchsize,1))+h1.dot(rbm.w.T))
				h2 = sigm(np.tile(rbm.c,(batchsize,1))+v2.dot(rbm.w))
				
				print(h1.shape)
				print(v1.shape)
				c1 = h1.dot(v1)
				c2 = h2.dot(v2)
				
				rbm.vw = rbm.momentum * rbm.vw + rbm.alpha * (c1 - c2).T / batchsize
				rbm.vb = rbm.momentum * rbm.vb + rbm.alpha * sum(v1 - v2) / batchsize
				rbm.vc = rbm.momentum * rbm.vc + rbm.alpha * sum(h1 - h2) / batchsize
				
				rbm.w = rbm.w + rbm.vw
				rbm.b = rbm.b + rbm.vb
				rbm.c = rbm.c + rbm.vc
				
				err = err + sum(sum((v1-v2)*(v1-v2)))/batchsize
				
	def dbntrain(self,X,opts=(1,0,100,2)):
		# n = 1;
		# x = train_x，60000个样本，每个维度为784，即60000*784
		#n为dbn中有几个rbm，这里n=2
		n = len(self.rbm);
		#充分训练第一个rbm
		self.rbmtrain(self.rbm[0], X, opts);
		#通过第一个rbm，依次训练后续的rbm
		for i in range(1,n):
			#建立rbm
			X = rbmup(self.rbm[i-1], X);
			#训练rbm
			self.rbmtrain(self.rbm[i], X, opts);
	
	def dbnunfoldtonn(self,layers):
		nueralNetwork = nn.neuralNetwork(layers);  
		#把每一层展开后的Weight拿去初始化NN的Weight  
		#注意dbn.rbm{i}.c拿去初始化了bias项的值  
		for i in range(len(layers)-1):
			nueralNetwork.weights[i] = [self.rbm[i].w]
			#nueralNetwork.weights[i].append(self.rbm[i].w)
		return nueralNetwork
	
def train():
	dbn1 = dbn([100,10],784)
	images = mt.loadImageSet(1)
	labels = mt.loadLabelSet(1)
	print(images.shape)
	dbn1.dbntrain(images)
	nn = dbn1.dbnunfoldtonn([784,100,10])
	nn.fit(images,labels)
	nn.predict(images[0])
train()