import numpy as np
import cv2

im = cv2.imread('.\\images\\doge.jpg')

line = np.asarray(np.reshape(im, (1, 256*256, 3)))

zeros = np.zeros((2, 3, 5))

z = np.asarray(np.reshape(zeros, (3, 2, 5)))

print(type(line))
print(zeros, '\n!!!\n', z)