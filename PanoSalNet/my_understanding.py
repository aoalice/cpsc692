import cv2
import numpy as np
#import caffe
import timeit
import os

FILE_IMAGE = '../Comparison/inputs/VST/topic1_21.png'
filename = '~/Comparison/inputs/VST/topic1_21.png'

img = cv2.imread(FILE_IMAGE)
mu = [ 104.00698793, 116.66876762, 122.67891434]
input_scale = 0.0078431372549

H = 9
W = 16

dat = cv2.resize(img, (W, H))

#print(dat)

dat = dat.astype(np.float32)
#dat -= np.array(mu)#mean values
#dat *= input_scale
#dat = dat.transpose((2,0,1))
print(dat.shape, img.shape)
cv2.imwrite("dat-skier.png", dat)
numericalData = np.asarray(dat[:,:,0])
print(numericalData)
print("final shape: ", numericalData.shape)
