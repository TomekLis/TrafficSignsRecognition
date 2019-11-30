#from model_version_1 import get_model as get_model_1
#from model_version_2 import get_model as get_model_2
from model_3 import cnn_model
from train_model import train_model
from get_data import get_mixed_data, get_separate_data
from plot_history import plot_history
import numpy as np

# model1 = get_model_1()
# trained_model = get_model_2()
trained_model = cnn_model()

# (train_X, test_X, train_y, test_y) = get_mixed_data()
(train_X, test_X, train_y, test_y) = get_separate_data()

history = train_model(trained_model, train_X, train_y, test_X, test_y, 15)

plt = plot_history(history, 15)

savePlot = input("Save plot?(y/n)")

yesString = "y"
if savePlot == yesString:
    plt.savefig("model_plot.png")

saveModel = input("Save model?(y/n)")

if saveModel == yesString:
    trained_model.save("traffic_sign_full_training_set_3.model")