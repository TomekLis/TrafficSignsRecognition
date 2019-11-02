import matplotlib.pyplot as plt
import csv
import cv2
import os
IMG_SIZE = 70


def createSignsTrainingData(rootpath):
    training_data = []
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            # img_array = cv2.imread(prefix + row[0], cv2.IMREAD_GRAYSCALE)
            # training_data.append([plt.imread(prefix + row[0]), row[7]])
            img_array = cv2.imread(prefix + row[0], cv2.IMREAD_GRAYSCALE)  # convert to array
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
            training_data.append([new_array, row[7]])  # add this to our training_data

            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels, training_data

def createSignsTestingData(rootpath):
    training_data = []
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    gtFile = open(rootpath + '/' + 'GT-final_test.test.csv') # annotations file
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    for row in gtReader:
        next(gtReader) # skip header
        # loop over all images in current annotations file
            # img_array = cv2.imread(prefix + row[0], cv2.IMREAD_GRAYSCALE)
            # training_data.append([plt.imread(prefix + row[0]), row[7]])
        img_array = cv2.imread(rootpath + '/' + row[0], cv2.IMREAD_GRAYSCALE)  # convert to array
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
        training_data.append([new_array, row[7]])  # add this to our training_data
       
        images.append(plt.imread(rootpath + '/' +row[0])) # the 1th column is the filename
        labels.append(row[7]) # the 8th column is the label
    gtFile.close()
    return images, labels, training_data
