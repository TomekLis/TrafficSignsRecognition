import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from readTrafficSigns import readTrafficSigns as rtf

# CATEGORIES = ["Dog", "Cat"]

images, labels = rtf('./GTSRB/Final_Training/Images')


def prepare(file_value):
    IMG_SIZE = 70
    img_array = images[file_value]
    img_array = img_array/255.0
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    plt.imshow(new_array)
    plt.show()
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("trafficsigns.model")

# prediction = model.predict([prepare('dog.jpg')])
# print(prediction)  # will be a list in a list.
# print(CATEGORIES[int(prediction[0][0])])


def predict_classes(model, images_test, labels_test):
    
    # Predict class of image using model
    class_pred = model.predict(images_test)

    # Convert vector to a label
    labels_pred = np.argmax(class_pred,axis=1)

    # Boolean array that tell if predicted label is the true label
    correct = (labels_pred == labels_test)

    # Array which tells if the prediction is correct or not
    # And predicted labels
    return correct, labels_pred

isCorrect, lables_pred = predict_classes(model, prepare(1000), 0)
print(isCorrect)
# print(CATEGORIES[int(lables_pred)])