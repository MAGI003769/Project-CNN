import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import types


# convert to gray scale

#im = Image.open('ba.PNG')
#im = im.convert('L')
#im.save('gray.PNG')

# gradient
im = cv2.imread('.\\images\\ba.PNG', 0)
sobelx = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=5)
sobely = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=5)
gradx = sobelx.ravel().transpose()
grady = sobely.ravel().transpose()
a1 = np.abs(gradx-grady)/np.sqrt(gradx**2+grady**2)
a2 = np.sqrt(2)*np.minimum(gradx, grady)/np.sqrt(gradx**2+grady**2)

plt.subplot(2, 4, 1), plt.title('x')
plt.imshow(sobelx, cmap = 'gray'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 2), plt.title('y')
plt.imshow(sobely, cmap = 'gray'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 3), plt.title('a1')
plt.imshow(a1.reshape([64, 64]), cmap = 'gray'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 4), plt.title('a2')
plt.imshow(a2.reshape([64, 64]), cmap = 'gray'), plt.xticks([]), plt.yticks([])
#plt.subplot(2, 4, 5), plt.title('-x')
#plt.imshow(-sobelx, cmap = 'gray'), plt.xticks([]), plt.yticks([])
#plt.subplot(2, 4, 6), plt.title('-y')
#plt.imshow(-sobely, cmap = 'gray'), plt.xticks([]), plt.yticks([])
plt.show()