import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import glob
import cv2
import os

from sklearn.model_selection import train_test_split
from skimage.color import rgb2grey

NUM_CLASSES = 43
classValue = 38
# path to the images
data_path = './GTSRB/Final_Training/Images'

images = []
image_labels = []

# get the image paths
image_path = data_path + '/' + format(classValue, '05d') + '/'
for img in glob.glob(image_path + '*.ppm'):
    image = cv2.imread(img)
    image = rgb2grey(image)
    image = (image / 255.0)  # rescale
    image = cv2.resize(image, (32, 32))  # resize
    images.append(image)
    # create the image labels and one-hot encode them
    labels = np.zeros((NUM_CLASSES, ), dtype=np.float32)
    labels[classValue] = 1.0
    image_labels.append(labels)

images = np.stack([img[:, :, np.newaxis]
                   for img in images], axis=0).astype(np.float32)
image_labels = np.matrix(image_labels).astype(np.float32)

plt.imshow(images[45, :, :, :].reshape(32, 32), cmap='gray')
plt.show()
print(image_labels[45, :])
