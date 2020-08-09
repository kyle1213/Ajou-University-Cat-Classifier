from keras.models import model_from_json, load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import Input
import matplotlib.pyplot as plt
import os
import numpy as np
import PIL.Image as pilimg
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils


"""
json_file = open("resnet.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("weights.h5")
"""
"""
loaded_model = load_model('resnet-50.h5')
"""

loaded_model = tf.keras.models.load_model('resnet.h5', custom_objects={'KerasLayer':hub.KerasLayer})
print("loaded model and weights")
loaded_model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

target = (224, 224)
image = pilimg.open(r'C:\Users\Admin\Desktop\deeplearning\ajou cat project resnet 18\test\test.jpg')
image = image.resize(target)
image = img_to_array(image)
image = np.expand_dims(image, axis = 0)
image = image/255
print("preprocessing Image...")


result = []
for i in range(10):
    result.append(0)
predict = loaded_model.predict(image)
predict = predict * 100
np.ndarray.tolist(predict)
for i in range(10):
    result[i] = str(predict[0][i])
    if result[i][2] == '.':
        result[i] = result[i][0:4]
    elif result[i][1] == '.':
        result[i] = result[i][0:3]
for i in range(5):
    answer = np.argmax(predict)
    if predict[0][answer] <= 0.1:
        break
    print("%d 위: "%(i+1), end='')
    
    if answer == 0:
        print("고등어 " + result[0] + "%")
    elif answer == 1:
        print("대목주 " + result[1] + "%")
    elif answer == 2:
        print("따봉이 " + result[2] + "%")
    elif answer == 3:
        print("방울이 " + result[3] + "%")
    elif answer == 4:
        print("삐약이 " + result[4] + "%")
    elif answer == 5:
        print("삼색이 " + result[5] + "%")
    elif answer == 6:
        print("점례 " + result[6]+ "%")
    elif answer == 7:
        print("챠밍이 " + result[7] + "%")
    elif answer == 8:
        print("코블이 " + result[8] + "%")
    elif answer == 9:
        print("학치 " + result[9] + "%")

    predict[0][answer] = 0








