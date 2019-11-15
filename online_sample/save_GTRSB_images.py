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
np.random.seed(42)

# path to the images
data_path = './GTSRB/Final_Training/Images'

images = []
image_labels = []

# get the image paths
for i in range(NUM_CLASSES):
    image_path = data_path + '/' + format(i, '05d') + '/'
    for img in glob.glob(image_path + '*.ppm'):
        image = cv2.imread(img)
        image = rgb2grey(image)
        image = (image / 255.0) # rescale
        image = cv2.resize(image, (32, 32)) #resize
        images.append(image)
        
        # create the image labels and one-hot encode them
        labels = np.zeros((NUM_CLASSES, ), dtype=np.float32)
        labels[i] = 1.0
        image_labels.append(labels)

images = np.stack([img[:, :, np.newaxis] for img in images], axis=0).astype(np.float32)
image_labels = np.matrix(image_labels).astype(np.float32)

plt.imshow(images[45, :, :, :].reshape(32, 32), cmap='gray')
plt.show()
print(image_labels[45, :])

print(images.shape)
print(len(images))

(train_X, test_X, train_y, test_y) = train_test_split(images, image_labels, 
                                                      test_size=0.2, 
                                                      random_state=42)
pickle_out = open("train_X.pickle","wb")
pickle.dump(train_X, pickle_out)
pickle_out.close()

pickle_out = open("train_y.pickle","wb")
pickle.dump(train_y, pickle_out)
pickle_out.close()

pickle_out = open("test_X.pickle","wb")
pickle.dump(test_X, pickle_out)
pickle_out.close()

pickle_out = open("test_y.pickle","wb")
pickle.dump(test_y, pickle_out)
pickle_out.close()
