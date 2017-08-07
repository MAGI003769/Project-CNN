import os
import numpy as np
import cv2
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat('train2data.mat')
#test_data = sio.loadmat('new_my2data.mat')

print('Original Data shape: ', data['pover'].shape)
#print(test_data['pover'].shape)


#for i in range(test_data['pover'].shape[0]):
#	print(test_data['pover'][i, -1, :])

#sio.savemat('new_my3data.mat', {'pover': test_data['pover']})

train_x = np.asarray([np.reshape(x, (192,192,10), order='F') for x in data['pover'][:, 0:-1, :]])
train_y = np.reshape(data['pover'][:, -1, 0], (data['pover'].shape[0]))

print('Shape of data into CNN: ', train_x.shape)
print('Shape of each elements in data feed into CNN: ', train_x[0].shape)
print('Shape of label into CNN: ', train_y.shape)
print('check the dimension of some selected element: ', data['pover'][:, -1, 0].shape[0])

select = 12

for i in range(10):
	plt.subplot(2, 5, i+1)
	plt.imshow(train_x[select, :, :, i])
print(train_y[select])
plt.show()