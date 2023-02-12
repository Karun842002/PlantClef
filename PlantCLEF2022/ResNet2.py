import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
import os

print(tf.config.list_physical_devices(
    device_type='GPU'
))

train_path="/home/jansi/plantclef/PlantCLEF2022/train/images"
test_path="/home/jansi/plantclef/PlantCLEF2022/val/images"
class_names=os.listdir(train_path)
class_names_test=os.listdir(test_path)
n=len(class_names)
input_shape=(224,224,3)
IMG_HEIGHT = 224
IMG_WIDTH = 224
with tf.device('/device:GPU:1'):
	train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_path, labels="inferred", label_mode="categorical",batch_size=32,image_size=(224,224),shuffle=True)
	val_ds = tf.keras.preprocessing.image_dataset_from_directory(test_path, labels="inferred", label_mode="categorical",batch_size=32,image_size=(224,224))
	restnet = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
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

	history = model.fit(train_ds,validation_data = val_ds, steps_per_epoch = 500, epochs=30, validation_steps=50, verbose=1)
	model.save('restnet.h5')
	print('Done')
