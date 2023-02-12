from gc import callbacks
import os
import tensorflow as tf
import json
from tqdm import tqdm 
test_path = '/home/jansi/plantclef/PlantCLEF2022/test/test/images'
model_path = '/home/jansi/models'

def load_image(path, image_size, num_channels=3):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img,expand_animations = False)
    img = tf.image.resize(img, image_size)
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img

#test_ds = tf.keras.utils.image_dataset_from_directory(test_path, label_mode = None, image_size=(224,224), batch_size = 1)
preds = []
with tf.device('/GPU:1'):
	model = tf.keras.models.load_model(os.path.join(model_path,'root_root_to_class.h5'))
	for i in tqdm(range(1,55307)):
		img = load_image(os.path.join(test_path,str(i)+'.jpg'),(224,224),3)
		img = tf.keras.preprocessing.image.img_to_array(img)
		img = img.reshape(1,224,224,3)
		pred = tf.argmax(model(img),1).numpy()[0]
		preds.append(pred)
file.open('list.txt','w').write(str(preds))
