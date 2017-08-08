import os
import numpy as np
import cv2
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat('train1data.mat')
#test_data = sio.loadmat('testdata.mat')

# ------- the following paragraph is to read multi files -------- #
"""
train_x = []
train_y = []
for i in range(3):
	data = sio.loadmat('train'+str(i+1)+'data.mat')
	train_x = train_x + [np.reshape(x, (192,192,10), order='F') for x in data['pover'][:, 0:-1, :]]
	train_y = train_y + [x for x in data['pover'][:, -1, 0]]
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
"""
print('Original Data shape: ', data['pover'].shape)
#print(test_data['pover'].shape)


train_x = np.asarray([np.reshape(x, (192,192,10), order='F') for x in data['pover'][:, 0:-1, :]])
train_y = np.reshape(data['pover'][:, -1, 0], (data['pover'].shape[0]))

print('Shape of data into CNN: ', train_x.shape)
print('Shape of each elements in data feed into CNN: ', train_x[0].shape)
print('Shape of label into CNN: ', train_y.shape)
print('check the dimension of some selected element: ', data['pover'][:, -1, 0].shape)

select = 36

for i in range(10):
	plt.subplot(2, 5, i+1)
	plt.imshow(train_x[select, :, :, i])
	plt.axis('off')
print(train_y[select])
plt.show()