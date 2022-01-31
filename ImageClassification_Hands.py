# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#### Importations des bibliothèques
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory ##new

#### Importation des images
os.chdir("/Users/mathisagathe/Desktop/M2/Machine learning")

# Génération de deux datasets : train et validation
image_size = (180, 180)
image_height = 180
image_width = 180
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Hands",
    validation_split=0.2, # 20% pour la validation et 80 pour la formation
    subset="training",
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Hands",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

# Noms des classes
class_names = train_ds.class_names
print(class_names)

#### Configurer les données pour une meilleure performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



#### Standardisation des données
normalization_layer = layers.Rescaling(1./255)
# NB : Inclure la couche de normalisation dans le modèle simplifie son déploiement

#### Creation du modèle
num_classes = 6

####: Essai d'augmentation des données via plusieurs modifications d'images :
    # Inversion horizontale
    # Rotation de 20°
    # Constraste aléatoire de l'image
    # Translation de morceaux aléatoires de l'image
#Non utilisé dans le modèle final


#data_augmentation = tf.keras.Sequential([
  #tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'), ## flip image
  #tf.keras.layers.experimental.preprocessing.RandomRotation(0.2), ## rotate 20o anlge
  #tf.keras.layers.RandomContrast(factor = 0.5),
  #tf.keras.layers.RandomCrop(height = 150, width = 150),
  #tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2, fill_mode="wrap"),
#])


## Répétitions sur plusieurs images et affichages des images modifiées
#for image, _ in train_ds.take(1):
 # plt.figure(figsize=(10, 10))
  #first_image = image[0]
  #for i in range(9):
   # ax = plt.subplot(3, 3, i + 1)
    #augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    #plt.imshow(augmented_image[0] / 255)
    #plt.axis('off')
    
#Modèle
    
model = Sequential([
    #redimensionnement des images
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(image_height, image_width, 3)),
    #Convolution 1
    layers.Conv2D(16, 4, padding='same', activation='relu'),
    layers.MaxPool2D(),
    #Convolution 2
    layers.Conv2D(32, 4, padding='same', activation='relu'),
    layers.MaxPool2D(),
    #Convolution 3
    layers.Conv2D(64, 4, padding='same', activation='relu'),
    layers.MaxPool2D(),
    #Drop out afin de réduire le surapprentissage
    layers.Dropout(0.2),
    layers.Flatten(),
    #Connection
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])


#### Compilation du modèle
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


#### Entraînement du modèle
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

#### Visualisation des résultats
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
 
#Graphique affichant les courbes de training et validation accury et loss
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#### Prédiction sur une nouvelle image
img = keras.preprocessing.image.load_img(
    "handtest.jpg", target_size=image_size)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) 

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)














