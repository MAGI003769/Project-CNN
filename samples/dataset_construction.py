import os
import numpy as np
import cv2
import pandas as pd
import scipy

dataset = np.empty((1, 64*64+1))

for char in range(10):
    for channel in range(9):
        if channel == 0:
            im_dir = ".\\picture" + str(char+1) + "\\g.jpg"
        else:
            im_dir = ".\\picture" + str(char+1) + "\\picture_" + str((channel-1)*45) + ".jpg"
        im = cv2.imread(im_dir)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = np.asarray(im)
        im_width = im.shape[0]
        im_height = im.shape[1]
        im = im.reshape([1, im_width*im_height])
        im = np.c_[im, char]
        dataset = np.r_[dataset, im]
    pass
pass
df = pd.DataFrame(dataset[1:-1, :])
df.to_csv('samples.csv', header = 0)

data = pd.read_csv("samples.csv", header = 0)
samples = np.asarray(data.ix[:,0:-2])
for i in range(samples.shape[0]/9):


