import tensorflow as tf
import json
import os
names = ["class_Cycadopsida_to_species.h5",
         "family_Goodeniaceae_to_species.h5",
         "class_Ginkgoopsida_to_species.h5",
         "family_Griseliniaceae_to_species.h5",
         "class_Gnetopsida_to_species.h5",
         "family_Grubbiaceae_to_species.h5",
         "class_Pinopsida_to_species.h5",
         "family_Gyrostemonaceae_to_species.h5",
         "family_Achatocarpaceae_to_species.h5",
         "family_Heliotropiaceae_to_species.h5",
         "family_Aizoaceae_to_species.h5",
         "family_Hydrangeaceae_to_species.h5",
         "family_Akaniaceae_to_species.h5",
         "family_Hydrophyllaceae_to_species.h5",
         "family_Alseuosmiaceae_to_species.h5",
         "family_Hydrostachyaceae_to_species.h5",
         "family_Amaranthaceae_to_species.h5",
         "family_Kewaceae_to_species.h5",
         "family_Amborellaceae_to_species.h5",
         "family_Limeaceae_to_species.h5",
         "family_Anacampserotaceae_to_species.h5",
         "family_Limnanthaceae_to_species.h5",
         "family_Anisophylleaceae_to_species.h5",
         "family_Loasaceae_to_species.h5",
         "family_Apiaceae_to_species.h5",
         "family_Menyanthaceae_to_species.h5",
         "family_Apodanthaceae_to_species.h5",
         "family_Molluginaceae_to_species.h5",
         "family_Aquifoliaceae_to_species.h5",
         "family_Montiaceae_to_species.h5",
         "family_Araliaceae_to_species.h5",
         "family_Moringaceae_to_species.h5",
         "family_Argophyllaceae_to_species.h5",
         "family_Namaceae_to_species.h5",
         "family_Asteropeiaceae_to_species.h5",
         "family_Nepenthaceae_to_species.h5",
         "family_Basellaceae_to_species.h5",
         "family_Nyctaginaceae_to_species.h5",
         "family_Begoniaceae_to_species.h5",
         "family_Nyssaceae_to_species.h5",
         "family_Boraginaceae_to_species.h5",
         "family_Parnassiaceae_to_species.h5",
         "family_Brassicaceae_to_species.h5",
         "family_Physenaceae_to_species.h5",
         "family_Bruniaceae_to_species.h5",
         "family_Phytolaccaceae_to_species.h5",
         "family_Buxaceae_to_species.h5",
         "family_Pittosporaceae_to_species.h5",
         "family_Cactaceae_to_species.h5",
         "family_Plumbaginaceae_to_species.h5",
         "family_Calyceraceae_to_species.h5",
         "family_Polygonaceae_to_species.h5",
         "family_Campanulaceae_to_species.h5",
         "family_Portulacaceae_to_species.h5",
         "family_Canellaceae_to_species.h5",
         "family_Resedaceae_to_species.h5",
         "family_Capparaceae_to_species.h5",
         "family_Rousseaceae_to_species.h5",
         "family_Cardiopteridaceae_to_species.h5",
         "family_Salvadoraceae_to_species.h5",
         "family_Caricaceae_to_species.h5",
         "family_Sarcobataceae_to_species.h5",
         "family_Caryophyllaceae_to_species.h5",
         "family_Schisandraceae_to_species.h5",
         "family_Celastraceae_to_species.h5",
         "family_Staphyleaceae_to_species.h5",
         "family_Ceratophyllaceae_to_species.h5",
         "family_Stegnospermataceae_to_species.h5",
         "family_Chloranthaceae_to_species.h5",
         "family_Stemonuraceae_to_species.h5",
         "family_Cleomaceae_to_species.h5",
         "family_Stixaceae_to_species.h5",
         "family_Columelliaceae_to_species.h5",
         "family_Stylidiaceae_to_species.h5",
         "family_Cordiaceae_to_species.h5",
         "family_Talinaceae_to_species.h5",
         "family_Cornaceae_to_species.h5",
         "family_Tamaricaceae_to_species.h5",
         "family_Crossosomataceae_to_species.h5",
         "family_Tropaeolaceae_to_species.h5",
         "family_Didiereaceae_to_species.h5",
         "family_Winteraceae_to_species.h5",
         "family_Droseraceae_to_species.h5",
         "order_Acorales_to_species.h5",
         "family_Ehretiaceae_to_species.h5",
         "root_root_to_class.h5",
         "family_Frankeniaceae_to_species.h5"]

train_path = '/home/jansi/plantclef/PlantCLEF2022/train/images'

def load_image(path, image_size, num_channels=3):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size)
    img = tf.reshape(img, (image_size[0], image_size[1], num_channels))
    return img


def create_dataset(image_paths, image_size, num_channels, label):
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(
        [label for i in range(len(image_paths))])
    img_ds = path_ds.map(lambda x: load_image(
        x, image_size, num_channels), num_parallel_calls=tf.data.AUTOTUNE)
    img_ds = tf.data.Dataset.zip((img_ds, label_ds)).shuffle(2000).take(2000)
    return img_ds


for nam in names:
    ROOT_LEVEL, ROOT_NODE, _, CLASSIFICATION_LEVEL = nam.split('.')[0].split('_')
    split = json.load(open('split.json'))
    class_names = set()
    for x in split.values():
        if x[CLASSIFICATION_LEVEL] not in class_names:
            class_names.add(x[CLASSIFICATION_LEVEL])
    class_names = list(class_names)
    n = len(class_names)
    class_to_idx = {class_names[i]: i for i in range(n)}

    image_paths = [[] for i in range(n)]
    for species in os.listdir(train_path):
        if not str(species).isnumeric():
            continue
        if ROOT_LEVEL != 'root':
            if split[species][ROOT_LEVEL] != ROOT_NODE:
                continue
        for fname in os.listdir(os.path.join(train_path, species)):
            label = class_to_idx[split[species][CLASSIFICATION_LEVEL]]
            image_path = os.path.join(train_path, species, fname)
            image_paths[label].append(image_path)
    print('Testing and Creating Test Dataset')
    train_ds_parts = [create_dataset(
        image_paths[i], (224, 224), 3, i) for i in range(n) if len(image_paths[i]) != 0]
    print(len(train_ds_parts))
    if(len(train_ds_parts) == 0):
        continue
    for i in range(1, len(train_ds_parts)):
        train_ds_parts[0] = train_ds_parts[0].concatenate(train_ds_parts[i])
    train_ds = train_ds_parts[0].shuffle(len(train_ds_parts[0]))
    print('TRAIN:Found', len(train_ds), 'images in',
          len(train_ds_parts), 'classes')
    if(len(train_ds) < 30):
        continue
    if(len(train_ds) > 20000):
        train_ds = train_ds.take(20000)
    labels = []
    for row in train_ds:
        if row[1] not in labels:
            labels.append(row[1])
    open(nam.split('.')[0]+'.txt','w').write(str(labels))
