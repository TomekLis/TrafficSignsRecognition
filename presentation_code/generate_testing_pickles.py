import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import glob 
import cv2
import os
import csv

from skimage.color import rgb2grey

NUM_CLASSES = 43

# path to the images
data_path = './GTSRB/Final_Test/Images'

images = []
image_labels = []


gtFile = open(data_path + '/' + 'GT-final_test.test.csv') # annotations file
gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
for row in gtReader:
    next(gtReader) 
    image = cv2.imread(data_path + '/' + row[0])
    image = rgb2grey(image)
    image = (image / 255.0) # rescale
    image = cv2.resize(image, (32, 32)) #resize
    images.append(image)

    labels = np.zeros((NUM_CLASSES, ), dtype=np.float32)
    class_value = int(row[7])
    print(row)
    labels[class_value] = 1.0
    image_labels.append(labels) 
        
images = np.stack([img[:, :, np.newaxis] for img in images], axis=0).astype(np.float32)
image_labels = np.matrix(image_labels).astype(np.float32)

pickle_out = open("test_images.pickle","wb")
pickle.dump(images, pickle_out)
pickle_out.close()

pickle_out = open("test_labels.pickle","wb")
pickle.dump(image_labels, pickle_out)
pickle_out.close()
