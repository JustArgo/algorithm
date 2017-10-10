'''
	本算法 用于学习 朴素贝叶斯
	1 读取文本并得到dict数据
	2 公式p(y.i/x) = p(x/y.i)*p(y.i)/p(x)
	3 p(y.i/x) = p(a1/y.i)*p(a2/y.i)*p(a3/y.i)*p(y.i)/p(x)
	
'''

import os
import codecs
import random

#变量 categories = []

categories = []
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
        for i in range(0,1):
            filename = os.path.join('%s\%s' % (child, files[i]))
            fopen = codecs.open(filename)
            data = fopen.read()
            global total_words
            total_words = total_words | set(data.split())
            fopen.close()
        #print child.decode('gbk') # .decode('gbk')是解决中文显示乱码问题
        #print(child)


# 读取文件内容并打印
def readFile(filename):
    fopen = open(filename, 'r') # r 代表read
    for eachLine in fopen:
        print("读取到得内容如下：",eachLine)
    fopen.close()
	
eachFile("C:\\Users\\Administrator\\Desktop\\python\\20_newsgroups")
#print(categories)
#print(total_words)
total_words = (b'the',b'many',b'to',b'and',b'news',b'have',b'for',b'not',b'as',b'of')
#print('------------')
#print(total_words)
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
all_word_dict = {}
for category in categories:
	word_dict = countWord("C:\\Users\\Administrator\\Desktop\\python\\20_newsgroups\\"+category)
	global all_word_dict
	all_word_dict[category] = word_dict

probability = {}

'''
	从20个中随机抽取一个文档 并计算 指定的10个单词的概率 分母 p(x) = p(the) * p(many) * p(to) ...
'''
def calRandomDocProbability():
	doc_dict = {} #随机选中的文档 对应 10个单词的 count字典
	total_count = 0
	total_probability = 1.0
	news_type = random.randint(0,19)# 产生一个随机数 0-19 代表 从20个新闻组中 随机选取一个
	files = os.listdir("C:\\Users\\Administrator\\Desktop\\python\\20_newsgroups\\"+categories[news_type])# 列出选定的新闻组里面的1000个文档
	random_id = random.randint(500,999) #代表从后面的501到1000中 选取一个文档
	for word in total_words:
		doc_dict[word] = 0
	for word in total_words:
		for text in open("C:\\Users\\Administrator\\Desktop\\python\\20_newsgroups\\"+categories[news_type]+"\\"+files[random_id]).read().split():
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
	
print(probability)




	