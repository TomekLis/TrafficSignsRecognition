import numpy as np
import matplotlib.pyplot as plt

def plot_history(history, epochs):
    num_epochs = np.arange(0, epochs)
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
    return plt
