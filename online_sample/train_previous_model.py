import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import glob 
import cv2
import os
import pickle
import time

pickle_in = open("train_X.pickle","rb")
train_X = pickle.load(pickle_in)

pickle_in = open("train_y.pickle","rb")
train_y = pickle.load(pickle_in)

pickle_in = open("test_X.pickle","rb")
test_X = pickle.load(pickle_in)

pickle_in = open("test_y.pickle","rb")
test_y = pickle.load(pickle_in)

model = tf.keras.models.Sequential()
input_shape = (32, 32, 1) # grey-scale images of 32x32

model.add(tf.keras.layers.Conv2D(32, (5, 5), padding='same', 
            activation='relu', input_shape=input_shape))
model.add(tf.keras.layers.BatchNormalization(axis=-1))      
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.2))
        
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.Dense(43, activation='softmax'))

optimizer = tf.keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
              metrics=['accuracy'])

history = model.fit(train_X, train_y, 
                   validation_data=(test_X, test_y),
                   epochs=10)

num_epochs = np.arange(0, 10)
plt.figure(dpi=300)
plt.plot(num_epochs, history.history['loss'], label='train_loss', c='red')
plt.plot(num_epochs, history.history['val_loss'], 
    label='val_loss', c='orange')
plt.plot(num_epochs, history.history['accuracy'], label='train_acc', c='green')
plt.plot(num_epochs, history.history['val_accuracy'], 
    label='val_acc', c='blue')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()
plt.savefig("previous_version.png")
# model.save('traffic_signs_previous.model')
