import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import glob 
import cv2
import os
from skimage.color import rgb2grey

NUM_CLASSES = 43

# path to the images
data_path = './GTSRB/Final_Training/Images'

images = []
image_labels = []

# get the image paths
for i in range(NUM_CLASSES):
    image_path = data_path + '/' + format(i, '05d') + '/'
    for img in glob.glob(image_path + '*.ppm'):
        image = cv2.imread(img)
        images.append(image)
        # create the image labels and one-hot encode them
        break


# images = np.stack([img[:, :, np.newaxis] for img in images], axis=0).astype(np.float32)

plt.imshow(images[42], cmap='gray')
plt.show()