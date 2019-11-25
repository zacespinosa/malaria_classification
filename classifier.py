from __future__ import absolute_import, division, print_function, unicode_literals

"""
We used the https://www.tensorflow.org/tutorials/images/classification tutorial, 
the CS231N course notes on CNNs and CS279 course notes on image processing in order
to design and implement this project.
"""

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import pipeline
import numpy as np
import matplotlib.pyplot as plt

batch_size = 128
IMG_SIZE = 128
epochs = 15


train_dataset = pipeline.Datasets('Train')
validation_dataset = pipeline.Datasets('Validation', name='validation')

# Generator for our training data; normalize images
train_image_generator = ImageDataGenerator(rescale=1./255)
# # Generator for our validation data; normalize images
validation_image_generator = ImageDataGenerator(rescale=1./255) 


train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dataset.get_dir(),
                                                           shuffle=True,
                                                           target_size=(IMG_SIZE, IMG_SIZE),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dataset.get_dir(),
                                                              target_size=(IMG_SIZE, IMG_SIZE),
                                                              class_mode='binary')

sample_training_images, _ = next(train_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# plotImages(sample_training_images[:5])

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_SIZE, IMG_SIZE ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=train_dataset.total_examples // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=validation_dataset.total_examples // batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

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