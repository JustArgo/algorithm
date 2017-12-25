'''
	
	1 有输入， 训练集有label
	2 输入是3维的，则神经网络的输入层为 3个神经元  
	3 mnist输出是 10个 神经元，识别 0 - 9 
	

'''



import numpy as np
 
# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
 
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
 
# output dataset            
y = np.array([[0,0,1,1]]).T
 
# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)
 
# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1
#print(syn0)
 
for iter in xrange(10):
	# forward propagation
	l0 = X
	l1 = nonlin(np.dot(l0,syn0))
	#print(l1.shape)
	#exit()
	# how much did we miss?
	l1_error = y - l1
	#print(l1_error.shape)
	# multiply how much we missed by the 
	# slope of the sigmoid at the values in l1
	#print(type(l1))
	l1_delta = l1_error * nonlin(l1,True)
	print(l1_delta)

	# update weights
	syn0 += np.dot(l0.T,l1_delta)
print "Output After Training:"
print l1