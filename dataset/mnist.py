import numpy as np
import struct

path = 'C:/Users/Administrator/Desktop/python/algorithm/dataset/'
def loadImageSet(which=0):
    binfile=None
    if which==0:
        binfile = open(path+"train-images.idx3-ubyte", 'rb')
    else:
        binfile=  open(path+"t10k-images.idx3-ubyte", 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>IIII' , buffers ,0)

    offset=struct.calcsize('>IIII')
    imgNum=head[1]
    width=head[2]
    height=head[3]
    #[60000]*28*28
    bits=imgNum*width*height
    bitsString='>'+str(bits)+'B' #like '>47040000B'

    imgs=struct.unpack_from(bitsString,buffers,offset)

    binfile.close()
    imgs=np.reshape(imgs,[imgNum,width*height])
    return imgs

def loadLabelSet(which=0):
    binfile=None
    if which==0:
        binfile = open(path+"train-labels.idx1-ubyte", 'rb')
    else:
        binfile=  open(path+"t10k-labels.idx1-ubyte", 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>II' , buffers ,0)
    imgNum=head[1]

    offset = struct.calcsize('>II')
    numString='>'+str(imgNum)+"B"
    labels= struct.unpack_from(numString , buffers , offset)
    binfile.close()
    labels=np.reshape(labels,[imgNum,1])

    #print labels
    return labels
'''
if __name__=="__main__":
    imgs=loadImageSet()
    #import PlotUtil as pu
    #pu.showImgMatrix(imgs[0])
    loadLabelSet()
'''