# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import datasets, layers, models
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

pickle_in = open("X_training.pickle","rb")
X_training = pickle.load(pickle_in)

pickle_in = open("y_training.pickle","rb")
y_training = pickle.load(pickle_in)

X = X/255.0
X_training = X_training/255.0

model = models.Sequential()
model.add(layers.Conv2D(70, (3, 3), input_shape=X.shape[1:]))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X, y, epochs=10, 
                    validation_data=(X_training, y_training))

test_loss, test_acc = model.evaluate(X_training, y_training, verbose=2)

model.save('traffic_signs.model')

print(test_acc)

# dense_layers = [0]
# layer_sizes = [64]
# conv_layers = [3]

# for dense_layer in dense_layers:
#     for layer_size in layer_sizes:
#         for conv_layer in conv_layers:
#             NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
#             print(NAME)

#             model = Sequential()

#             model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
#             model.add(Activation('relu'))
#             model.add(MaxPooling2D(pool_size=(2, 2)))

#             for l in range(conv_layer-1):
#                 model.add(Conv2D(layer_size, (3, 3)))
#                 model.add(Activation('relu'))
#                 model.add(MaxPooling2D(pool_size=(2, 2)))

#             model.add(Flatten())

#             for _ in range(dense_layer):
#                 model.add(Dense(layer_size))
#                 model.add(Activation('relu'))

#             model.add(Dense(1))
#             model.add(Activation('sigmoid'))

#             tensorboard = TensorBoard(log_dir="logs\\{}".format(NAME))

#             model.compile(loss='binary_crossentropy',
#                           optimizer='adam',
#                           metrics=['accuracy'],
#                           )

#             model.fit(X, y,
#                       batch_size=32,
#                       epochs=10,
#                       validation_split=0.3,
#                       callbacks=[tensorboard])

model.save('trafficsigns.model')