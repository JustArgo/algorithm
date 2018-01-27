from numpy import *
class cluster_node:
	def __init__(self,vec,left=None,right=None,distance=0.0,id=None,count=1):
		self.left = left
		self.right = right
		self.vec = vec
		self.id = id
		self.distance = distance
		self.count = count
		
def L2dist(v1,v2):
	return sqrt(sum((v1-v2)**2))

def L1dist(v1,v2):
	return sum(abs(v1-v2))
	
def hcluster(features,distance=L2dist):
	distances = {}
	currentClusterId = -1
	
	clust = [cluster_node(array(features[i]),id=i) for i in range(len(features))]
	
	while len(clust)>1:
		lowestpair = (0,1)
		closest = distance(clust[0].vec,clust[1].vec)
		for i in range(len(clust)):
			for j in range(i+1,len(clust)):
				if (clust[i].id,clust[j].id) not in distances:
					distances[(clust[i].id,clust[j].id)] = distance(clust[i].vec,clust[j].vec)
				
				d = distances[(clust[i].id,clust[j].id)]
				
				if d<closest:
					closest = d
					lowestpair = (i,j)
		
		
		mergevec = [(clust[lowestpair[0]].vec[i] + clust[lowestpair[1]].vec[i]) / 2.0 for i in range(len(clust[0].vec))]
				
		newcluster = cluster_node(array(mergevec),left=clust[lowestpair[0]],right=clust[lowestpair[1]],distance=closest,id=currentClusterId)		
		
		currentClusterId += 1
		del clust[lowestpair[0]]		
		del clust[lowestpair[1]]
		clust.append(newcluster)
		
	return clust[0]
	

					
					
					
					