import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
import os

print(tf.config.list_physical_devices(
    device_type='GPU'
))

train_path="/data/images"
test_path="/data/images"
class_names=os.listdir(train_path)
class_names_test=os.listdir(test_path)
n=len(class_names)
input_shape=(224,224,3)
IMG_HEIGHT = 224
IMG_WIDTH = 224
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(train_path,target_size=(224, 224),batch_size=450,shuffle=True,class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_path,target_size=(224,224),batch_size=450,shuffle=False,class_mode='categorical')
restnet = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(224,224,3))
output = restnet.layers[-1].output
output = tf.keras.layers.Flatten()(output)
restnet = tf.keras.Model(restnet.input, outputs=output)
for layer in restnet.layers:
    layer.trainable = False
restnet.summary()

model = tf.keras.Sequential()
model.add(restnet)
model.add(tf.keras.layers.Dense(512, activation='relu', input_dim=input_shape))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(n, activation='sigmoid'))
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
model.summary()

history = model.fit(train_generator, 
                    steps_per_epoch = 1000, 
                    epochs=30,
                    validation_steps=50, 
                    verbose=1)

model.save('restnet.h5')

