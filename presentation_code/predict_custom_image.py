import cv2
from skimage.color import rgb2grey
import numpy as np
import matplotlib.pyplot as plt

labelNames = ["Speed limit (20km/h)"
,"Speed limit (30km/h)"
,"Speed limit (50km/h)"
,"Speed limit (60km/h)"
,"Speed limit (70km/h)"
,"Speed limit (80km/h)"
,"End of speed limit (80km/h)"
,"Speed limit (100km/h)"
,"Speed limit (120km/h)"
,"No passing"
,"No passing for vehicles over 3.5 metric tons"
,"Right-of-way at the next intersection"
,"Priority road"
,"Yield"
,"Stop"
,"No vehicles"
,"Vehicles over 3.5 metric tons prohibited"
,"No entry"
,"General caution"
,"Dangerous curve to the left"
,"Dangerous curve to the right"
,"Double curve"
,"Bumpy road"
,"Slippery road"
,"Road narrows on the right"
,"Road work"
,"Traffic signals"
,"Pedestrians"
,"Children crossing"
,"Bicycles crossing"
,"Beware of ice/snow"
,"Wild animals crossing"
,"End of all speed and passing limits"
,"Turn right ahead"
,"Turn left ahead"
,"Ahead only"
,"Go straight or right"
,"Go straight or left"
,"Keep right"
,"Keep left"
,"Roundabout mandatory"
,"End of no passing"
,"End of no passing by vehicles over 3.5 metric tons"]


def predict_custom_image(model, name):
    # inputImage = input("Type in image name")
    # image = cv2.imread(inputImage + '.png')
    image = cv2.imread(name)
    image = rgb2grey(image)
    image = cv2.resize(image, (32, 32))

    plt.imshow(image, cmap='gray')
    plt.show()

    img = np.expand_dims(image, axis=0)
    img = np.expand_dims(img, axis=3)

    class_pred = model.predict(img)

    # Convert vector to a label
    labels_pred = np.argmax(class_pred,axis=1)
    print("Predicted class " + labelNames[labels_pred[0]])
    
    # index_max = np.argmax(labels_test)

    # Boolean array that tell if predicted label is the true label
    # correct = (labels_pred == index_max)

    # Array which tells if the prediction is correct or not
    # And predicted labels
    return labels_pred
