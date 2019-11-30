def train_model(model, train_X, train_y, test_X, test_y, epochs_num):
    history = model.fit(train_X, train_y, 
                   validation_data=(test_X, test_y),
                   epochs=epochs_num)

    return history