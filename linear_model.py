from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pipeline
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
IMG_SIZE = 128
NUM_COMPONENTS = 512
EPOCHS = 20
batch_size = 128
TRAIN_EXAMPLES = 26560

def reshape_and_flatten(imgs): 
    """
    Reshape, Normalize, and Flatten Image
    """
    new_imgs = []
    for img in imgs:
        # Reshape image and preserve aspect ratio
        old_size = img.shape[:2]
        ratio = float(IMG_SIZE)/max(old_size)
        new_size = tuple([int(dim*ratio) for dim in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]))
        delta_w = IMG_SIZE - new_size[1]
        delta_h = IMG_SIZE - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        color = [0, 0, 0]
        new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)

        # Flatten Image
        new_img = new_img.flatten()
        new_imgs.append(new_img)

    return new_imgs 

def apply_pca(train, val):
    pca = PCA(n_components=NUM_COMPONENTS)
    new_train = pca.fit_transform(train)
    new_val = pca.transform(val)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()

    Visualize Difference
    for i in range(10): visualize(i, train, pca, new_train)

    return new_train, new_val, pca

"""
In order to visualize the PCA images we used the following tutorial: 
https://www.kaggle.com/pmmilewski/pca-decomposition-and-keras-neural-network
"""
def visualize(i, train, pca, new_train):
    org = train[i].reshape(IMG_SIZE,square_size, 3)
    inv_train = pca.inverse_transform(new_train)
    rec = inv_train[i].reshape(IMG_SIZE, square_size, 3)
    pair = np.concatenate((org, rec), axis=1)
    plt.figure(figsize=(4,2))
    plt.imshow(pair)
    plt.show()

def create_linear_model(): 
    model = Sequential([
        Dense(256, activation='relu'),
        Dropout(.2),
        Dense(128, activation='relu'),
        Dropout(.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model

def run_linear_model(model, train_gen, train_dataset, pca, val_data, val_labels):
    num_batches = TRAIN_EXAMPLES // batch_size
    for i in range(EPOCHS):
        tot_loss = 0
        tot_acc = 0 
        for j in range(num_batches):
            batch_data, batch_labels = next(train_gen)
            print('Starting batch ', j, '/', num_batches)
            if batch_data.shape[0] != batch_size: continue # skip last batch
            # Apply Filters to Batch 
            batch_data = pipeline.preprocess_batch(['gaussianHP', 'gaussian'], batch_data)
            # Flatten
            batch_data = batch_data.reshape(128, 128*128*3)
            # Reduce
            batch_data = pca.transform(batch_data)
            
            loss, acc = model.train_on_batch(batch_data, batch_labels) # , epochs=EPOCHS, batch_size=32, validation_split=.10)
            tot_loss += loss
            tot_acc += acc

        # Calculate average loss and accuracy in epoch
        print((tot_loss / num_batches), (tot_acc / num_batches))

        # Test Model after each epoch
        test_linear_model(model, val_data, val_labels)

def labels_and_shuffle(infected, uninfected):
    labels = np.concatenate((np.zeros(len(infected)), np.ones(len(uninfected))))
    data = np.concatenate((infected, uninfected))
    data, labels = shuffle(data, labels)
    return data, labels
    
def test_linear_model(model, val_data, val_labels): 
    val_loss, val_acc = model.evaluate(val_data,  val_labels, verbose=2)
    print('\nTest accuracy:', val_acc)

dim_reduction = {
    "pca": apply_pca,
}


def main(reduction):
    # Get datasets
    datasets = pipeline.Datasets(load=True, num_examples=1000)
    train_dataset = datasets.train
    validation_dataset = datasets.validation

    # Create dataset
    train_infected_flatten = reshape_and_flatten(train_dataset.imgs["Infected"])
    train_uninfected_flatten = reshape_and_flatten(train_dataset.imgs["Uninfected"])
    train_data, train_labels = labels_and_shuffle(train_infected_flatten, train_uninfected_flatten)

    val_infected_flatten = reshape_and_flatten(validation_dataset.imgs["Infected"])
    val_uninfected_flatten = reshape_and_flatten(validation_dataset.imgs["Uninfected"])
    val_data, val_labels = labels_and_shuffle(val_infected_flatten, val_uninfected_flatten)

    # Apply PCA 
    train_data, val_data, pca = apply_pca(train_data, val_data)

    # Generator for our training data; normalize images
    train_image_generator = ImageDataGenerator(rescale=1./255)

    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dataset.get_dir(),
                                                           shuffle=True,
                                                           target_size=(IMG_SIZE, IMG_SIZE),
                                                           class_mode='binary')

    # Train and Test Model
    model = create_linear_model()
    run_linear_model(model, train_data_gen, train_dataset, pca, val_data, val_labels)


if len(sys.argv) > 2: 
    print("Pick one dim reduction method: pca ...")

main(sys.argv[1])