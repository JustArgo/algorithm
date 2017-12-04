import numpy as np
list = [[2.0,6.0],[4.1,12.1],[6.3,18.0]]
covMat = np.cov(np.array(list,dtype=float),rowvar=0)
print(covMat)
a,b = np.linalg.eig(covMat)
print(a)
print(np.argsort(a))


