import numpy as np
def splitData(dataArr):
	length = len(dataArr)
	trainDataLength = int(length * 0.8)
	trainDataArr = []
	testDataArr = []
	indexSet = set()
	while len(indexSet) < trainDataLength:
		index = np.random.randint(length);
		indexSet.add(index)
		
	for i in range(length):
		if i in indexSet:
			trainDataArr.append(dataArr[i])
		else:
			testDataArr.append(dataArr[i])
	return trainDataArr,testDataArr
		