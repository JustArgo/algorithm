'''
	朴素贝叶斯理论
	1 主要概念 计算
	防止有些向量的值为0 
		p0Num = ones(numWords)
		p1Num = ones(numWords)
		
	防止 小数过小 下溢出
	p1Vect = log(p1Num/p1Denom)
	p0Vect = log(p0Num/p0Denom)
	
	//
	def classifyNB(vec2classify, p0Vec, p1Vec, pClass1):
		p1 = sum(vec2Classify * p1Vec) + log(pClass1)
		p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
		if p1 > p0:
			return 1
		else:
			return 0
	
	p0Vec 是 0分类中   各个单词 占用比例  
'''