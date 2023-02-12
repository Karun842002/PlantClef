# %%
import tensorflow as tf
from PIL import Image
import pandas as pd
import json
import os

# %%
train_path = '/home/jansi/plantclef/PlantCLEF2022/train/images'
val_path = '/home/jansi/plantclef/PlantCLEF2022/val/images'
CLASSIFICATION_LEVEL = 'class'
ROOT_NODE = 'root'
ROOT_LEVEL = 'root'
MODEL_NAME = ROOT_LEVEL+'_'+ROOT_NODE+'_to_'+CLASSIFICATION_LEVEL

# %%
def load_image(path, image_size, num_channels=3):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img,expand_animations = False)
    img = tf.image.resize(img, image_size)
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img

# %%
def create_dataset(image_paths, image_size, num_channels, label):
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices([label for i in range(len(image_paths))])
    img_ds = path_ds.map(lambda x: load_image(x, image_size, num_channels), num_parallel_calls=tf.data.AUTOTUNE)
    img_ds = tf.data.Dataset.zip((img_ds, label_ds)).shuffle(2000).take(2000)
    return img_ds

# %%
split = json.load(open('split.json'))
class_names = set()
for x in split.values(): 
    if x[CLASSIFICATION_LEVEL] not in class_names: class_names.add(x[CLASSIFICATION_LEVEL])
class_names = list(class_names)
n = len(class_names)
class_to_idx = {class_names[i]:i for i in range(n)}

# %%
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
        

# %%
train_ds_parts = [create_dataset(image_paths[i],(224,224),3,i) for i in range(n) if len(image_paths[i])!=0]
for i in range(1,len(train_ds_parts)): train_ds_parts[0] = train_ds_parts[0].concatenate(train_ds_parts[i])
train_ds = train_ds_parts[0].shuffle(2000*n)
print('Found',len(train_ds),'images in',len(train_ds_parts),'classes')
train_ds = train_ds.batch(32)

# %%
image_paths = [[] for i in range(n)]
for species in os.listdir(val_path):
    if not str(species).isnumeric(): continue
    if ROOT_LEVEL!='root':
        if split[species][ROOT_LEVEL]!=ROOT_NODE:
            continue
    for fname in os.listdir(os.path.join(train_path,species)):
        label=class_to_idx[split[species][CLASSIFICATION_LEVEL]]
        image_path = os.path.join(train_path,species,fname)
        image_paths[label].append(image_path)

# %%
val_ds_parts = [create_dataset(image_paths[i],(224,224),3,i) for i in range(n) if len(image_paths[i])!=0]
for i in range(1,len(val_ds_parts)): val_ds_parts[0] = val_ds_parts[0].concatenate(val_ds_parts[i])
val_ds = val_ds_parts[0].shuffle(2000*n)
print('Found',len(val_ds),'images in',len(val_ds_parts),'classes')
train_ds = val_ds.batch(32) 

# %%
model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3), classes = n)
output = model.layers[-1].output
output = tf.keras.layers.Flatten()(output)
model = tf.keras.Model(model.input, outputs=output)
for layer in model.layers: layer.trainable = False
model.summary()

# %%
m = tf.keras.Sequential()
m.add(model)
m.add(tf.keras.layers.Dense(512, activation='relu', input_dim=(224,224)))
m.add(tf.keras.layers.Dropout(0.3))
m.add(tf.keras.layers.Dense(512, activation='relu'))
m.add(tf.keras.layers.Dropout(0.3))
m.add(tf.keras.layers.Dense(n, activation='sigmoid'))
m.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
m.summary()

# %%
m.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=2e-5),
              metrics=['accuracy'])

# %%
history = m.fit(train_ds, epochs=30, validation_data=val_ds ,validation_steps=50, verbose=1)
m.save(MODEL_NAME+'.h5')
