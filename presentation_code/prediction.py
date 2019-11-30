from tensorflow.keras import models
from predict_custom_image import predict_custom_image

model = models.load_model("traffic_sign_full_training_set_3.model")

print(predict_custom_image(model, 'STOP_sign.jpg'))
print(predict_custom_image(model, '14.png'))
print(predict_custom_image(model, '00068.png'))
print(predict_custom_image(model, '36.png'))
print(predict_custom_image(model, '00053.png'))
print(predict_custom_image(model, '00068.png'))
print(predict_custom_image(model, '00218.png'))