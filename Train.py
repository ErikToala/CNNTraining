import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import matplotlib.pyplot as plt
import keras
from keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout, Activation 
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import numpy as np

K.clear_session()


labels = ["Aluminum","Cardboard","Ceramics","Crystal","Mud","Natural","Paper","Plastic","Unicel","Wood"]
img_size = 120

data_train = './Dataset/Category'
data_test = './Dataset/Test'

datagen = ImageDataGenerator(
   rescale=1./255, 
   shear_range=0.3,
   zoom_range=0.3,
   rotation_range = 30,
   horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)


imagen_train = datagen.flow_from_directory(
    data_train,
    target_size=(img_size,img_size),
    batch_size=10,
    class_mode='categorical'
)


imagen_test = test_datagen.flow_from_directory(
    data_test,
    target_size=(img_size,img_size),
    batch_size=2,
    class_mode='categorical'
)

model = Sequential()
model.add(Conv2D(32, (3,3), padding ="same", input_shape=(img_size, img_size, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64, (2,2), padding ="same"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(labels), activation='softmax'))

opt = Adam(lr=0.00005)
model.compile(loss='categorical_crossentropy',optimizer= opt,metrics=['accuracy'])


model.fit_generator(
    imagen_train, 
    steps_per_epoch=20, 
    epochs=20, 
    validation_data=imagen_test,
    #validation_steps=5
)


target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('./modelo/structureModelo.h5')
model.save_weights('./modelo/layerWeights.h5')