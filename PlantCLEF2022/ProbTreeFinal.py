import tensorflow as tf
from PIL import Image
import pandas as pd
import json
import os
import cv2
from tqdm import tqdm

def load_image(path, image_size, num_channels=3):
	img = tf.io.read_file(path)
	img = tf.image.decode_jpeg(img,channels=3,expand_animations = False)
	img = tf.image.resize(img, image_size)
	img = tf.reshape(img,(image_size[0], image_size[1], num_channels))
	return img

def clean(image_paths):
	new_image_paths = []
	for path in tqdm(image_paths):
		try:
			img = cv2.imread(path)
			shape = img.shape
		except:
			if os.path.exists(path): os.remove(path)
			continue
		new_image_paths.append(path)
	return new_image_paths

def create_dataset(image_paths, image_size, num_channels, label):
	#image_paths = clean(image_paths)
	path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
	label_ds = tf.data.Dataset.from_tensor_slices([label for i in range(len(image_paths))])
	img_ds = path_ds.map(lambda x: load_image(x, image_size, num_channels), num_parallel_calls=tf.data.AUTOTUNE)
	img_ds = tf.data.Dataset.zip((img_ds, label_ds)).shuffle(2000).take(2000)
	return img_ds

def log(message):
    f = open('logs.txt','a')
    f.write(message)
    f.close()

def train(data):
	with tf.device('/GPU:1'):
		tf.keras.backend.clear_session()
		train_path = '/home/jansi/plantclef/PlantCLEF2022/train/images'
		val_path = '/home/jansi/plantclef/PlantCLEF2022/val/images'
		level, root, root_level = data
		CLASSIFICATION_LEVEL = level
		ROOT_LEVEL = root_level
		ROOT_NODE = root
		MODEL_NAME = ROOT_LEVEL+'_'+ROOT_NODE+'_to_'+CLASSIFICATION_LEVEL
		VALIDATE = True
		split = json.load(open('split.json'))
		class_names = set()
		for x in split.values(): 
				if x[CLASSIFICATION_LEVEL] not in class_names: class_names.add(x[CLASSIFICATION_LEVEL])
		class_names = list(class_names)
		n = len(class_names)
		class_to_idx = {class_names[i]:i for i in range(n)}
		
		image_paths = [[] for i in range(n)]
		for species in os.listdir(train_path):
			if not str(species).isnumeric(): continue
			if ROOT_LEVEL!='root':
				if split[species][ROOT_LEVEL]!=ROOT_NODE:
					continue
			for fname in os.listdir(os.path.join(train_path,species)):
				label=class_to_idx[split[species][CLASSIFICATION_LEVEL]]
				image_path = os.path.join(train_path,species,fname)
				image_paths[label].append(image_path)
		print('Testing and Creating Test Dataset')
		train_ds_parts = [create_dataset(image_paths[i],(224,224),3,i) for i in range(n) if len(image_paths[i])!=0]
		print(len(train_ds_parts))
		for i in range(1,len(train_ds_parts)): train_ds_parts[0] = train_ds_parts[0].concatenate(train_ds_parts[i])
		train_ds = train_ds_parts[0].shuffle(len(train_ds_parts[0]))
		print('TRAIN:Found',len(train_ds),'images in',len(train_ds_parts),'classes')
		train_ds = train_ds.batch(1)
		
		image_paths = [[] for i in range(n)]
		for species in os.listdir(val_path):
			if not str(species).isnumeric(): continue
			if ROOT_LEVEL!='root':
				if split[species][ROOT_LEVEL]!=ROOT_NODE:
					continue
			for fname in os.listdir(os.path.join(val_path,species)):
				label=class_to_idx[split[species][CLASSIFICATION_LEVEL]]
				image_path = os.path.join(val_path,species,fname)
				image_paths[label].append(image_path)
		print('Testing and Creating Val Dataset')
		val_ds_parts = [create_dataset(image_paths[i],(224,224),3,i) for i in range(n) if len(image_paths[i])!=0]
		for i in range(1,len(val_ds_parts)): val_ds_parts[0] = val_ds_parts[0].concatenate(val_ds_parts[i])
		val_ds = val_ds_parts[0].shuffle(len(val_ds_parts[0]))
		print('VAL:Found',len(val_ds),'images in',len(val_ds_parts),'classes')
		val_ds = val_ds.batch(1) 
		print('Building Model')
		model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3), classes = n)
		output = model.layers[-1].output
		output = tf.keras.layers.Flatten()(output)
		model = tf.keras.Model(model.input, outputs=output)
		for layer in model.layers: layer.trainable = False
		model.summary()

		m = tf.keras.Sequential()
		m.add(model)
		m.add(tf.keras.layers.Dense(512, activation='relu', input_dim=(224,224)))
		m.add(tf.keras.layers.Dropout(0.3))
		m.add(tf.keras.layers.Dense(512, activation='relu'))
		m.add(tf.keras.layers.Dropout(0.3))
		m.add(tf.keras.layers.Dense(n, activation='sigmoid'))
		es = tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
		if(VALIDATE and len(val_ds)>=10):
			mc = tf.keras.callbacks.ModelCheckpoint(MODEL_NAME+'weights.{epoch:02d}-{val_accuracy:.2f}-{val_loss:.2f}.hdf5')
		else:
			mc = tf.keras.callbacks.ModelCheckpoint(MODEL_NAME+'weights.{epoch:02d}-{accuracy:.2f}-{loss:.2f}.hdf5')
		m.compile(loss='sparse_categorical_crossentropy',
				optimizer=tf.keras.optimizers.RMSprop(learning_rate=2e-5),
				metrics=['accuracy'])
		m.summary()
		if(VALIDATE and len(val_ds)>=10):
			history = m.fit(val_ds, epochs=25, validation_data=train_ds, validation_steps=50, verbose=1,callbacks=[es,mc])
		else:
			history = m.fit(train_ds, epochs=10, verbose=1, callbacks=[mc])
		m.save(MODEL_NAME+'.h5')

if __name__ == '__main__':
    models = [('class','root','root'),
	      ('species','Cycadopsida','class'),
              ('species','Ginkgoopsida','class'),
              ('species','Gnetopsida','class'),
              ('species','Lycopodiopsida','class'),
              ('order','Magnoliopsida','class'),
              ('species','Pinopsida','class'),
              ('order','Polypodiopsida','class')]

    hier = json.load(open('hier.json'))
    for order,vo in hier["Magnoliopsida"][1].items():
        models.append(('family',order,'order'))
        for family,vf in  vo[1].items():
            models.append(('species',family,'family'))
    for order,vo in hier["Polypodiopsida"][1].items():
        models.append(('species',order,'order'))
    n = len(models)
    for x in range(5,len(models)):
        log("Started "+str(x+1)+" of "+str(n))
        train(models[x])
        log("Finished "+str(x+1)+" of "+str(n))
