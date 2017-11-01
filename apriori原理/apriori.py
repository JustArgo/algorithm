'''	
	算法原理(用于寻找频繁项集)
	1 生成候选项集
'''
#加载数据
def loadDataSet():
	return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]
	
#扫描数据集 返回所有子项集合
def createC1(dataSet):
	C1 = []
	for transaction in dataSet:
		for item in transaction:
			if not [item] in C1:
				C1.append([item])
				
	C1.sort()
	return list(map(frozenset,C1))
	
#扫码所有项目，计算每个项的值，过滤项目中数目<最小支持度的数据
# 先计数 再过滤
# D是一个 集合 ([1,3,4],[2,3,5],[1,2,3,5],[2,5])
# 返回
#	retList 是过滤之后的项 的 key  例如上面以0.5过滤  则剩下 1 2 3 5      4的支持度只有 0.25
#	supportData 是 1 2 3 4 5 每个项的支持度
def scanD(D,Ck,minSupport):
	ssCnt = {}
	for tid in D:
		for can in Ck:
			if can.issubset(tid):
				if not can in ssCnt:
					ssCnt[can] = 1
				else:
					ssCnt[can] += 1
	numItems = float(len(D))
	retList = []
	supportData = {}
	for key in ssCnt:
		support = ssCnt[key]/numItems
		if support >= minSupport:
			retList.insert(0,key)
		supportData[key] = support
	return retList,supportData
	


#aprioriGen  根据Lk集合 以及k 生成 新的集合
#传  1 2 3 5  则返回  12 13 15 23 25 35
#
#
def aprioriGen(Lk,k):
	#print('Lk:',Lk)
	#print('k:',k)
	retList = []
	lenLk = len(Lk)
	#print('lenLk:',lenLk)
	for i in range(lenLk):
		for j in range(i+1,lenLk):
			L1 = list(Lk[i])[:k-2]   #代表的是 索引上限  第一次是0 所以为[] 之所以为2 举例如下  [1 2 3 4]  [1 2 3 5] 长度为4 传k=5 5-2=3   限制索引 2 必须做到前3位相同
			L2 = list(Lk[j])[:k-2]
			#print('list1:',list(Lk[i])[:k-2])
			#print('list2:',list(Lk[j])[:k-2])
			L1.sort()
			L2.sort()
			#print('L1:',L1)
			#print('L2:',L2)
			#print('i:',i)
			#print('j:',j)
			#print(L1==L2)
			if L1==L2:
				retList.append(Lk[i]|Lk[j])
	return retList
	
#apriori算法
def apriori(dataSet,minSupport=0.5):
	C1 = createC1(dataSet)
	D = list(map(set,dataSet))
	L1,supportData = scanD(D,C1,minSupport)
	L = [L1]
	k = 2
	while (len(L[k-2])>0):
		Ck = aprioriGen(L[k-2],k)
		#print('Ck:',Ck)
		Lk,supK = scanD(D,Ck,minSupport)
		#print('Lk:',Lk)
		#print('supK:',supK)
		supportData.update(supK)
		L.append(Lk)
		k += 1
	return L,supportData


	
#挖掘关联规则 输入的L  是  [ [{1} {2} {3} {5}] [{2,5} {3,5}] [{2,3,5}] ] 这样的集合 i=1的时候  每个元素的 规则系数 都为1
def generateRules(L,supportData,minConf=0.7):
	bigRuleList = []
	for i in range(1,len(L)):
		for freqSet in L[i]:
			H1 = [frozenset([item]) for item in freqSet]
			print('H1:',H1)
			if i>1:
				rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
			else:
				calcConf(freqSet,H1,supportData,bigRuleList,minConf)
	return bigRuleList
	
#对规则进行评估
def calcConf(freqSet, H, supportData, brl, minConf=0.5):
	prunedH = []
	for conseq in H:
		print('conseq:',conseq)
		conf = supportData[freqSet]/supportData[freqSet-conseq]
		print(freqSet-conseq,'-->',conseq,' conf:',conf)
		if conf >= minConf:
			#print(freqSet-conseq,'-->',conseq,' conf:',conf)
			brl.append((freqSet-conseq,conseq,conf))
			prunedH.append(conseq)
	return prunedH
	
#生成候选规则集合
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.5):
	m = len(H[0])
	if len(freqSet) > (m+1):
		Hmp1 = aprioriGen(H, m+1)
		Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
		print('Hmp1:',Hmp1)
		if len(Hmp1) > 1:
			rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

#加载训练数据
def loadTrainDataSet(trainFileName,separator='\t'):
	fopen = open(trainFileName)
	lineArr = []
	for line in fopen.readlines():
		lineArr.append(line.strip().split(separator))
	return lineArr

#加载测试数据
def loadTestDataSet(testFileName,separator='\t'):
	fopen = open(testFileName)
	lineArr = []
	for line in fopen.readlines():
		lineList = line.strip().split(separator)
		lineArr.append(map(float,lineList))
	return array(lineArr)
	
def apr(trainFileName='trainSet.txt'):
	#执行算法
	dataSet = loadTrainDataSet(trainFileName,',')
	C1 = createC1(dataSet)
	D = list(map(set,dataSet))
	
	#0.5 最小支持度 代表所占的比例 现实数据中 主要看中业务需求 
	#L1,suppData0 = scanD(D,C1,0.5)
	
	#L是什么   supportData是什么
	L,supportData = apriori(dataSet,0.5)
	print('L:',L)
	
	rules = generateRules(L,supportData,0.5)
	print('-----------------------------\n\n')
	print(rules)


