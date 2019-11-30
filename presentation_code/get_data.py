import pickle
from sklearn.model_selection import train_test_split

def get_mixed_data():
    pickle_in = open("training_images.pickle","rb")
    training_images = pickle.load(pickle_in)

    pickle_in = open("training_labels.pickle","rb")
    training_labels = pickle.load(pickle_in)

    (train_X, test_X, train_y, test_y) = train_test_split(training_images, training_labels, 
                                                      test_size=0.2, 
                                                      random_state=42)

    return train_X, test_X, train_y, test_y

def get_separate_data():
    pickle_in = open("training_images.pickle","rb")
    training_images = pickle.load(pickle_in)

    pickle_in = open("training_labels.pickle","rb")
    training_labels = pickle.load(pickle_in)

    pickle_in = open("test_images.pickle","rb")
    test_images = pickle.load(pickle_in)

    pickle_in = open("test_labels.pickle","rb")
    test_labels = pickle.load(pickle_in)


    return training_images, test_images, training_labels, test_labels
