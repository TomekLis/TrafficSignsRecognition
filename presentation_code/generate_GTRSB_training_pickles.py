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
np.random.seed(42)

# path to the images
data_path = './GTSRB/Final_Training/Images'

images = []
image_labels = []

# get the image paths
for i in range(NUM_CLASSES):
    image_path = data_path + '/' + format(i, '05d') + '/'
    print(str(i) + "/42")
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

pickle_out = open("training_images.pickle","wb")
pickle.dump(images, pickle_out)
pickle_out.close()

pickle_out = open("training_labels.pickle","wb")
pickle.dump(image_labels, pickle_out)
pickle_out.close()
