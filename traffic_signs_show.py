import matplotlib.pyplot as plt
from readTrafficSigns import readTrafficSigns as rtf

images, labels = rtf('./GTSRB/Final_Training/Images')

plt.imshow(images[1000])
plt.show()