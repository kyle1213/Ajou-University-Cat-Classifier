import  keras
from keras import models, layers
from keras import Input
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add
from tensorflow.python.client import device_lib
 
import os
import matplotlib.pyplot as plt
import numpy as np
import math

train_datagen = ImageDataGenerator(rescale = 1./255)

train_dir = os.path.join(r'C:\Users\Admin\Desktop\deeplearning\ajou cat project resnet 18\data\trainset')
 
train_generator = train_datagen.flow_from_directory(train_dir, batch_size=16, target_size=(224, 224), color_mode='rgb')

# number of classes
K = 10

input_tensor = Input(shape=(224, 224, 3), dtype='float32', name='input')
 
 
def conv1_layer(x):    
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
 
    return x   
 
def conv2_layer(x):
    shortcut = MaxPooling2D((2, 2), 2)(x)
 
    for i in range(2):
        if (i == 0):
            x = ZeroPadding2D(padding=(1, 1))(x)
            x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
        else:
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
 
            x = Add()([x, shortcut])   
            x = Activation('relu')(x)
    
    return x
 
def conv3_layer(x):
    shortcut = MaxPooling2D((2, 2), 2)(x)
    shortcut = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(shortcut)
    
    for i in range(2):     
        if(i == 0):
            x = ZeroPadding2D(padding=(1, 1))(x)
            x = Conv2D(128, (3, 3), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)           
            x = Activation('relu')(x)    
        
        else:
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])     
            x = Activation('relu')(x)
            
    return x

def conv4_layer(x):
    shortcut = MaxPooling2D((2, 2), 2)(x)
    shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)
  
    for i in range(2):     
        if(i == 0):
            x = ZeroPadding2D(padding=(1, 1))(x)
            x = Conv2D(256, (3, 3), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)               
        
        else:
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)
 
    return x

def conv5_layer(x):
    shortcut = MaxPooling2D((2, 2), 2)(x)
    shortcut = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(shortcut)
  
    for i in range(2):     
        if(i == 0):
            x = ZeroPadding2D(padding=(1, 1))(x)
            x = Conv2D(512, (3, 3), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)              
            x = Activation('relu')(x)                   
        
        else:
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)           
            
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)       
                  
    return x
 
x = conv1_layer(input_tensor)
x = conv2_layer(x)
x = conv3_layer(x)
x = conv4_layer(x)
x = conv5_layer(x)
x = GlobalAveragePooling2D()(x)
output_tensor = Dense(K, activation='softmax')(x)
 
resnet = Model(input_tensor, output_tensor)

resnet.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    
resnet.fit(train_generator, steps_per_epoch = 11, epochs = 50)


"""
resnet_json = resnet.to_json()
with open("resnet.json", "w") as json_file :
    json_file.write(resnet_json)
print("saved model architecture")
    
resnet.save_weights('weights.h5')
"""

resnet.save('resnet.h5')

print("saved weights")































