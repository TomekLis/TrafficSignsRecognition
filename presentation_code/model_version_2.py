import tensorflow as tf

def get_model():
    model = tf.keras.models.Sequential() # sequential API - used for 
    input_shape = (32, 32, 1) # grey-scale images of 32x32

    model.add(tf.keras.layers.Conv2D(32, (5, 5), padding='same', 
                activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))      
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(64, (5, 5), padding='same', 
                                     activation='relu'))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same', 
                                     activation='relu')) #
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

    return model
