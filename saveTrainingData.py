import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from create_signs_training_data import createSignsTrainingData, createSignsTestingData
import random
import pickle

IMG_SIZE = 70

images, labels, trainingData = createSignsTrainingData('./GTSRB/Final_Training/Images')
images_testing, labels_testing, trainingData_testing = createSignsTestingData('./GTSRB/Final_Test/Images')

X = []
y = []

for features,label in trainingData:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

X_training = []
y_training = []

for features,label in trainingData_testing:
    X_training.append(features)
    y_training.append(label)


X_training = np.array(X_training).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_training = np.array(y_training)

pickle_out = open("X_training.pickle","wb")
pickle.dump(X_training, pickle_out)
pickle_out.close()

pickle_out = open("y_training.pickle","wb")
pickle.dump(y_training, pickle_out)
pickle_out.close()
