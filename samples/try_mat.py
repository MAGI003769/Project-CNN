import os
import numpy as np
import cv2
import pandas as pd
import scipy.io as sio

x = np.random.rand(2, 3, 9)
y = np.random.rand(3, 2, 9)

sio.savemat('channels.mat', {'x': x})

data = sio.loadmat('channels.mat')



