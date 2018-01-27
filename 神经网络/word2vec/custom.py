'''
	自定义 word2vec
	word2vec在 
'''
from scipy.special import expit
from copy import deepcopy
import heapq
from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, fromstring, sqrt,\
    empty, sum as np_sum, ones, logaddexp
	
from six import iteritems, itervalues, string_types

#向量大小
vector_size = 20

#第一层大小
layer1_size = 20

#窗口大小
window = 5
	
class Vocab():
	def __init__(self,count=0,index=0,left=None,right=None):
		self.point = []
		self.code = []
		self.count = count
		self.index = index
		self.left = left
		self.right = right
		
	def __lt__(self, other):  # used for sorting in a priority queue
		return self.count < other.count

	
def seeded_vector(seed_string):
	once = random.RandomState(hash(seed_string) & 0xffffffff)
	return (once.rand(vector_size) - 0.5) / vector_size
	
	

fopen = open('fenci_result.txt',encoding='UTF-8')
countDict = {}
alpha = 0.001
wv_vocab = {}
word_vocabs = []
wv_index2word = []
for word in fopen:
	if word not in countDict:
		countDict[word] = 1
	else:	
		countDict[word] += 1
	
count=0	
for word in countDict:
	wv_vocab[word] = Vocab(count=countDict[word],index=count)
	word_vocabs.append(wv_vocab[word])
	wv_index2word.append(word)
	count += 1
print(countDict)
heap = list(itervalues(wv_vocab))
heapq.heapify(heap)
for i in range(len(wv_vocab) - 1):
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

#先得到 syn1[word.point]
syn1 = zeros((len(wv_vocab), layer1_size), dtype=REAL)


context_vectors = empty((count,vector_size),dtype=REAL)
for i in range(len(wv_vocab)):
	context_vectors[i] = seeded_vector(wv_index2word[i] + str(1))			
			
def train_sg_pair(word1,context_index):

	#代表单词2的上下文
	l1 = context_vectors[context_index]

	neu1e = zeros(l1.shape)

	predict_word = wv_vocab[word1]
	
	'''
	predict_word就是单词1的Vocab对象
	syn1[predict_word].point 代表当前计算得到的 单词1的 向量
	'''

	l2a = deepcopy(syn1[predict_word.point])

	prod_term = dot(l1,l2a.T)

	fa = expit(prod_term)

	#计算梯度误差
	ga = (1-predict_word.code-fa)*alpha

	#更新model.syn1[predict_word]
	syn1[predict_word.point] += outer(ga, l1)
			
			
for pos1,word1 in enumerate(word_vocabs):
	reduced_window = random.randint(window)
	start = max(0, pos1 - window + reduced_window)
	for pos2,word2 in enumerate(word_vocabs[start:(pos1+window+1-reduced_window)],start):
		if pos1!=pos2:
			train_sg_pair(wv_index2word[word1.index],pos2)
			
print(len(syn1))
		




