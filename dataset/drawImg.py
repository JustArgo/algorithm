import numpy as np     
import struct    
import matplotlib.pyplot as plt     
    
filename = 't10k-images.idx3-ubyte'    
#filename = 'C:/Users/haoming/Desktop/train-images-idx3-ubyte'   
binfile = open(filename,'rb')#以二进制方式打开    
buf = binfile.read()    
    
index = 0    
magic, numImages, numRows, numColums = struct.unpack_from('>IIII',buf,index)#读取4个32 int    
print (magic,' ',numImages,' ',numRows,' ',numColums  )  
index += struct.calcsize('>IIII')    
    
    
im = struct.unpack_from('>784B',buf,index)#每张图是28*28=784Byte,这里只显示第一张图    
index += struct.calcsize('>784B' )    
    
im = np.array(im)    
im = im.reshape(28,28)    
print( im )   
    
fig = plt.figure()    
plt.imshow(im,cmap = 'binary')#黑白显示    
plt.show() 