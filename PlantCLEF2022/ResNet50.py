import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
import os
import json

classification_level = "class"
print(tf.config.list_physical_devices(
	device_type='GPU'
))

table = tf.lookup.StaticHashTable(default_value=-1)

def parse_image(filename):
	f = filename.numpy()
	print(f)
	return
	parts = tf.strings.split(filename, os.sep)
	label = parts[-2]
	sl =  label.numpy()
	split = json.load(open('split.json'))
	label = split[sl][classification_level]
	if count[sl]>=2000: return
	count[sl] = count.get(sl,0) + 1
	image = tf.io.read_file(filename)
	image = tf.io.decode_jpeg(image)
	image = tf.image.convert_image_dtype(image, tf.float32)
	image = tf.image.resize(image, [224, 224])
	return image, label

with tf.device('/device:CPU:0'):
	train_path="/home/jansi/plantclef/PlantCLEF2022/train/images"
	test_path="/home/jansi/plantclef/PlantCLEF2022/val/images"
	class_names=os.listdir(train_path)
	list_ds = tf.data.Dataset.list_files(train_path+"/*/*")
	train_ds = list_ds.map(parse_image)
	val_list_ds = tf.data.Dataset.list_files(test_path+'/*/*')
	val_ds = val_list_ds.map(parse_image)
	# restnet = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(224,224,3))
	# output = restnet.layers[-1].output
	# output = tf.keras.layers.Flatten()(output)
	# restnet = tf.keras.Model(restnet.input, outputs=output)
	# for layer in restnet.layers:
	# 		layer.trainable = False
	# restnet.summary()

	# model = tf.keras.Sequential()
	# model.add(restnet)
	# model.add(tf.keras.layers.Dense(512, activation='relu', input_dim=input_shape))
	# model.add(tf.keras.layers.Dropout(0.3))
	# model.add(tf.keras.layers.Dense(512, activation='relu'))
	# model.add(tf.keras.layers.Dropout(0.3))
	# model.add(tf.keras.layers.Dense(n, activation='sigmoid'))
	# model.compile(loss='categorical_crossentropy',
	# 		  optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
	# 		  metrics=['accuracy'])
	# model.summary()

	# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	# 	filepath='/home/jansi/plantclef/PlantCLEF2022/checkpoint',
	# 	save_weights_only = True,
	# 	monitor = 'val_accuracy',
	# 	mode = 'max',
	# )

	# history = model.fit(train_ds,  
	# 				epochs=22,
	# 		steps_per_epoch =  1000,
	# 		validation_data = val_ds,
	# 				validation_steps=50, 
	# 				verbose=1,
	# 		callbacks=[model_checkpoint_callback]
	# 	   )

	# model.save('restnet.h5')
	# print('Model Saved')
