'''
	本算法 用于学习 朴素贝叶斯
	1 读取文本并得到dict数据
	2 公式p(y.i/x) = p(x/y.i)*p(y.i)/p(x)
	3 p(y.i/x) = p(a1/y.i)*p(a2/y.i)*p(a3/y.i)*p(y.i)/p(x)
	
'''

import os
import codecs
import random
import re
from numpy import *
import operator

#变量 categories = []

categories = []
docPath = "C:\\Users\\Administrator\\Desktop\\python\\20_newsgroups\\";
algorithmPath = "C:\\Users\\Administrator\\Desktop\\python\\algorithm\\"
trainDocCount = 10
# 遍历指定目录，显示目录下的所有文件名
total_words = set('')
def eachFile(filepath):
    pathDir =  os.listdir(filepath)
    global categories
    categories = pathDir
    for allDir in pathDir:
        #print(allDir)
        child = os.path.join('%s\%s' % (filepath, allDir))
        files = os.listdir(child)
        #print(files[:80])
        for i in range(trainDocCount):
            filename = os.path.join('%s\%s' % (child, files[i]))
            fopen = codecs.open(filename)
            data = fopen.read()
            global total_words
            total_words = total_words | set(data.split())
            fopen.close()
        #print child.decode('gbk') # .decode('gbk')是解决中文显示乱码问题
        #print(child)


#停用词过滤器 字典专用 
def stopWordsFilter4Dict(wordDataSet,stopWord):
	tmpDict = wordDataSet.copy()
	for key in tmpDict.keys():
		if key.lower() in stopWord:
			wordDataSet.pop(key)

#停用词过滤器 list set专用
def stopWordFilter4ListAndSet(wordDataSet,stopWord):
	tmpSet = wordDataSet.copy()
	for word in tmpSet:
		if word.lower() in stopWord:
			wordDataSet.remove(word)

#删除非单词字符
def noneVocaFilter(dataSet):
	tmpSet = dataSet.copy()
	if type(dataSet).__name__ == 'list':
		for i in range(len(tmpSet)):
			if re.match(r'^[a-zA-Z]*$',tmpSet[i]) == None:
				dataSet.remove(tmpSet[i])
				
	elif type(dataSet).__name__ == 'set':
		for word in tmpSet:
			if re.match(r'^[a-zA-Z]*$',word) == None:
				dataSet.remove(word)
		
	else:
		return
# 读取文件内容并打印
def readFile(filename):
    fopen = open(filename, 'r') # r 代表read
    for eachLine in fopen:
        print("读取到得内容如下：",eachLine)
    fopen.close()

# 将所有字符转换成小写
def toLower(dataSet):
	if type(dataSet).__name__ == 'list':
		tmpSet = []
		for i in range(len(dataSet)):
			tmpSet[i] = dataSet[i].lower()
		
	elif type(dataSet).__name__ == 'set':
		tmpSet = set('')
		for word in dataSet:
			tmpSet.add(word.lower())
		
	else:
		return
	return tmpSet
#bytes 转  str
def bytes2str(words):
	for word in words:
		if type(word).__name__ == 'bytes':
			words.remove(word)
			words.add(word.decode('iso-8859-1'))
			
#准备 停用词 用set效率更高
def prepareStopWords():
	stopWordList = set();
	stopWordFilePath = algorithmPath+"stop_words.txt"
	fopen = open(stopWordFilePath)
	for word in fopen.readlines():
		stopWordList.add(word.replace('\n',''))
	return stopWordList
		
#将文档转换成向量 特定方法 每个分类 10个文档 则 200个向量  200行 1831列  再返回一个labels 列表 代表这200个向量对应的分类   向量为2值型  有则表示为1 无则表示为0
def transferDoc2Vect(all_words):
	
	total_list = [] #二维列表 200行  1831列
	labels = []
	word_count_dict = {}
	#初始化字典
	for word in all_words:
		word_count_dict[word] = 0
	#1 先计算每个单词的数量 
	for category in categories:
		categoryPath = docPath + category
		files = os.listdir(categoryPath)
		for i in range(trainDocCount):
			wordSet = set(open(categoryPath + "\\" + files[i],encoding='iso-8859-1').read().split())
			bytes2str(wordSet)
			noneVocaFilter(wordSet)
			for word in wordSet:
				if word in all_words:
					word_count_dict[word] += 1
	sortedWordCount = sorted(word_count_dict.items(),key=operator.itemgetter(1),reverse=True)
	all_words = set('')
	for i in range(10):#排序在前的10个单词
		all_words.add(sortedWordCount[i][0])
	total_words_list = list(all_words)
	
	for category in categories:
		categoryPath = docPath + category
		files = os.listdir(categoryPath)
		for i in range(trainDocCount):
			vect = zeros(len(all_words))#初始化一个向量
			wordSet = set(open(categoryPath + "\\" + files[i],encoding='iso-8859-1').read().split())
			bytes2str(wordSet)
			noneVocaFilter(wordSet)
			for word in wordSet:
				if word in all_words:
					vect[total_words_list.index(word)] = 1
					break
			total_list.append(vect)
			labels.append(category)
	
	return total_list,labels
	
#根据返回的向量 和 标签 计算 每个分类的向量 
def calcTotalVect(vect,labels):
	total_vect = []
	for category in categories:
		category_vect = zeros(len(vect[0]))
		for i in range(len(vect)):
			if labels[i] == category:  #当前循环的向量 分类 为 category
				category_vect += vect[i]
		total_vect.append(category_vect)
	return total_vect
	
#根据分类向量 计算概率
def calcTotalProb(vect):
	total_prob = []
	for i in range(len(vect)):
		total_prob.append(vect[i]/sum(vect[i]))
	return total_prob
'''
eachFile("C:\\Users\\Administrator\\Desktop\\python\\20_newsgroups")
bytes2str(total_words) #将 bytes 转换成 str
noneVocaFilter(total_words)
total_words = toLower(total_words)
stopWords = prepareStopWords()
stopWordFilter4ListAndSet(total_words,stopWords)
vect,labels = transferDoc2Vect(total_words)
total_vect = calcTotalVect(vect,labels)
total_prob = calcTotalProb(total_vect)
print(total_vect)
print(total_prob)
exit()
'''
#少一步 给定篇文档  如何对其进行分类

				

def countWord(filepath):
	files =  os.listdir(filepath)
	word_dict = {}
	for word in total_words:
		word_dict[word] = 0
	for word in total_words:
		for i in range(0,10):
			for text in codecs.open(filepath + "\\" + files[i]).read().split():
				if word == text:
					word_dict[word] = word_dict[word] + 1
	return word_dict
#
'''
all_word_dict = {}
for category in categories:
	word_dict = countWord(docPath+category)
	global all_word_dict
	all_word_dict[category] = word_dict

probability = {}
'''


'''
	从20个中随机抽取一个文档 并计算 指定的10个单词的概率 分母 p(x) = p(the) * p(many) * p(to) ...
'''
def calRandomDocProbability():
	doc_dict = {} #随机选中的文档 对应 10个单词的 count字典
	total_count = 0
	total_probability = 1.0
	news_type = random.randint(0,19)# 产生一个随机数 0-19 代表 从20个新闻组中 随机选取一个
	files = os.listdir(docPath+categories[news_type])# 列出选定的新闻组里面的1000个文档
	random_id = random.randint(500,999) #代表从后面的501到1000中 选取一个文档
	for word in total_words:
		doc_dict[word] = 0
	for word in total_words:
		for text in open(docPath+categories[news_type]+"\\"+files[random_id]).read().split():
			if word.decode('utf-8') == text:
				print('%s == %s' % (word,text))
				total_count = total_count + 1
				doc_dict[word] = doc_dict[word] + 1
	for text in doc_dict:
		if doc_dict[text] != 0:
			total_probability = total_probability * (doc_dict[text] / total_count)
	return total_probability

'''
	有20个分类 所以要计算p(y.1/x) - p(y.20/x)
	p(y.1/x) = p() * p(y.1) / p('the','many')
'''
'''
for category in categories:
	total = 0
	percent = 1.0
	for word in all_word_dict[category]:
		total = total + all_word_dict[category][word]
	for word in all_word_dict[category]:
		percent = percent * (all_word_dict[category][word]/total)
	percent = percent * (1/20)
	px = calRandomDocProbability()
	global probability
	probability[category] = percent / px
'''	
#print(probability)


#加载训练数据
def loadTrainDataSet(trainFileName,separator='\t'):
	fopen = open(trainFileName)
	lineArr = []
	labelArr = []
	for line in fopen.readlines():
		lineList = line.strip().split(separator)
		featArr = []
		for i in range(len(lineList)-1):
			featArr.append(float(lineList[i]))
		lineArr.append(featArr)
		labelArr.append(lineList[len(lineList)-1])
	return array(lineArr),labelArr

def loadTestDataSet(testFileName,separator='\t'):
	fopen = open(testFileName)
	lineArr = []
	for line in fopen.readlines():
		lineList = line.strip().split(separator)
		lineArr.append(map(float,lineList))
	return array(lineArr)

	
#朴素贝叶斯 
def bayes(trainFileName,testFileName,separator='\t'):
	# 1 得到数据和分类
	# 2 得到分类和特征的对应关系字典 {'calssify1':[32,1,33,0,4]}  代表calssify1分类的 第一个特征有32个  占比 32/32+1+33+0+4
	# 3 按分类区分得到总概率字典，key为分类的名称，value为分类的概率， {'classify1':0.33,'calssify2':0.67}
	# 4 计算每个分类的每个特征的概率 {'calssify1':[0.3,0.2,0.1,0.3,0.1],'calssify2':[0.0,0.1,0.2,0.3,0.4]}
	# 5 输入一个新的数据，循环和每个分类的 特征list相乘并加起来，得到值，看哪个分类的值大，就是哪个分类
	lineArr,labels = loadTrainDataSet(trainFileName,separator)
	labelSet = set(labels)
	clasDict = {}
	clasProbabilityDict = {}
	featProbabilityDict = {}
	for clas in labelSet:
		clasList = []
		featProbList = []
		for i in range(len(labels)):
			if labels[i]==clas:
				clasList.append(lineArr[i])
		clasSum = sum(clasList,axis=0)
		clasDict[clas] = clasSum
		for featNum in clasSum:
			featProbList.append(featNum/sum(clasSum))
		featProbabilityDict[clas] = featProbList
		clasProbabilityDict[clas] = len(clasList)/float(len(labels))
	
	print('clasProbabilityDict->',clasProbabilityDict)
	print('clasDict->',clasDict)
	print('featProbabilityDict->',featProbabilityDict)
	testDataSet = loadTestDataSet(testFileName,separator)
	for line in testDataSet:
		probDict = {}
		for clas in labelSet:
			probDict[clas] = sum(line * featProbabilityDict[clas])*clasProbabilityDict[clas]
		probList = sorted(probDict.iteritems(),key=lambda p:p[1],reverse=True)
		print(probList[0][0])
		



	