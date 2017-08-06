import os
import numpy as np
import cv2
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat('my2data(1).mat')
#test_data = sio.loadmat('new_my2data.mat')

print(data['pover'].shape)
#print(test_data['pover'].shape)


#for i in range(test_data['pover'].shape[0]):
#	print(test_data['pover'][i, -1, :])

#sio.savemat('new_my3data.mat', {'pover': test_data['pover']})

train_x = np.asarray([np.reshape(x, (192,192,10)) for x in data['pover'][:, 0:-1, :]])
#test_x = np.asarray([np.reshape(x, (192,192,10)) for x in test_data['pover'][:, 0:-1, :]])
#train_y = np.reshape(data['pover'][:, -1, 0], (data['pover'].shape[0]))

#plt.subplot(121)
plt.imshow(train_x[12, :, :, 0])
#plt.subplot(122)
#plt.imshow(test_x[69, :, :, 0])
plt.show()

#print(train_x.shape)
#print(train_y.shape)