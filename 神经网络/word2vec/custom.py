'''
	自定义 word2vec
'''
from scipy.special import expit
from numpy import *
from copy import deepcopy
import heapq


class Vocab:
	def __init__(self,count=0,index=0,left,right):
		self.point = []
		self.code = []
		self.count = count
		self.index = index
		self.left = left
		self.right = right
	
fopen = open('fenci_result.txt')
countDict = {}
alpha = 0.001
wv_vocab = {}
for word in fopen:
	if word not in countDict:
		countDict[word] = 1
	else:	
		countDict[word] += 1
	
i=0	
for word in countDict:
	wv_vocab[word] = Vocab(count=countDict[word],index=i)
	i += 1

heap = list(itervalues(wv_vocab))
heapq.heapify(heap)
for i in xrange(len(wv_vocab) - 1):
	min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
	heapq.heappush(
		heap, Vocab(count=min1.count + min2.count, index=i + len(wv_vocab), left=min1, right=min2)
	)

# recurse over the tree, assigning a binary code to each vocabulary word
if heap:
	max_depth, stack = 0, [(heap[0], [], [])]
	while stack:
		node, codes, points = stack.pop()
		if node.index < len(wv_vocab):
			# leaf node => store its path from the root
			node.code, node.point = codes, points
			max_depth = max(len(codes), max_depth)
		else:
			# inner node => continue recursion
			points = array(list(points) + [node.index - len(wv_vocab)], dtype=uint32)
			stack.append((node.left, array(list(codes) + [0], dtype=uint8), points))
			stack.append((node.right, array(list(codes) + [1], dtype=uint8), points))

	
for pos1,word1 in enumerate(countDict):
	for pos2,word2 in enumerate(countDict[]):
		if pos1!=pos2:
			train_sg_pair(word1,pos2)
		
#先得到 syn1[word.point]


def train_sg_pair(word1,context_index):

	#代表单词2的上下文
	l1 = context_vectors[context_index]

	neu1e = zeros(l1.shape)

	predict_word = wv[word1]
	
	'''
	predict_word就是单词1的Vocab对象
	syn1[predict_word].point 代表当前计算得到的 单词1的 向量
	'''

	l2a = deepcopy(syn1[predict_word.point])

	prod_term = dot(l1,l2a)

	fa = expit(prod_term)

	#计算梯度误差
	ga = (1-predict_word.code-fa)*alpha

	#更新model.syn1[predict_word]
	syn1[predict_word.point] += outer(ga, l1)

