from tensorflow.keras import datasets, layers, models
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob 
from skimage.color import rgb2grey


value=1002

pickle_in = open("train_X.pickle","rb")
train_X = pickle.load(pickle_in)

pickle_in = open("train_y.pickle","rb")
train_y = pickle.load(pickle_in)

pickle_in = open("test_X.pickle","rb")
test_X = pickle.load(pickle_in)

pickle_in = open("test_y.pickle","rb")
test_y = pickle.load(pickle_in)


model = models.load_model("traffic_signs.model")
model.summary()

def prepare(img_value):
    plt.imshow(img_value.reshape(32, 32), cmap='gray')
    # plt.imshow(img_value, cmap='gray')
    plt.show()

    img = np.expand_dims(img_value, axis=0)
    # img = np.expand_dims(img, axis=3)
    return img


def predict_classes(model, images_test, labels_test):
    
    # Predict class of image using model
    class_pred = model.predict(images_test, batch_size=32)

    # Convert vector to a label
    labels_pred = np.argmax(class_pred,axis=1)
    
    index_max = np.argmax(labels_test)

    # Boolean array that tell if predicted label is the true label
    correct = (labels_pred == index_max)

    # Array which tells if the prediction is correct or not
    # And predicted labels
    return correct, labels_pred

image = cv2.imread('14.png')
image = rgb2grey(image)
image = (image / 255.0) # rescale
image = cv2.resize(image, (32, 32))

isCorrect, lables_pred = predict_classes(model, prepare(test_X[value]), test_y[value])
# isCorrect, lables_pred = predict_classes(model, prepare(image), 14)
# isCorrect, lables_pred = predict_classes(model, test_X[value], test_y[value])
print(isCorrect[0])
print(lables_pred[0])
# print(np.argmax(test_y[value]))
# print(class_names[int(lables_pred)])