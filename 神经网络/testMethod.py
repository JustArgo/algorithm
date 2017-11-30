import numpy as np
from math import *

def tanh(x):
	return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
	
list = [[1,2],[3,4],[5,6]]
list2 = [[2,3,1],[2,3,1],[2,3,1]]
print(np.add(list,list2))
