import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import types

from skimage.feature import hog
from skimage import data, color, exposure
# gradient
im = cv2.imread('.\\images\\lines.PNG', 0)
sobelx = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=5)
sobely = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=5)
#gradx = sobelx.ravel().transpose()
#grady = sobely.ravel().transpose()
mag, ang = cv2.cartToPolar(sobelx, sobely)
print('mag\n', mag)
print('ang\n', ang)

image = cv2.imread('.\\images\\origin.PNG', 0)

fd, hog_image = hog(image, orientations=4, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualise=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()
print(hog_image_rescaled.shape)