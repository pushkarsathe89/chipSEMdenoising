"""
Denoising SEM Images with Autoencoder and Image Augmentation

Authors:
    In-Ho Lee, KRISS, August 22, 2021
    Seoleun Shin, KRISS, 2023-Jan-01
References:
    - https://keras.io/examples/vision/oxford_pets_image_segmentation/
    - https://keras.io/examples/vision/autoencoder
"""

import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D
)
from tensorflow.keras.utils import img_to_array

# Set random seed for reproducibility
np.random.seed(456)
tf.random.set_seed(456)

# Constants
TARGET_WIDTH = 512
TARGET_HEIGHT = 512
IMAGE_SIZE = 512

# GPU/CPU configuration
def list_devices():
    from tensorflow.python.client import device_lib
    devices = device_lib.list_local_devices()
    print("Available devices:", devices)
    print("GPUs:", tf.config.list_physical_devices('GPU'))
list_devices()

# Clean up previous data
def clean_data_dirs():
    extensions = ['*.tif', '*.jpg']
    dirs = ['Data/time/low', 'Data/time/high', 'Data/test', 'Data/test_cleaned']
    for img_path in dirs:
        for ext in extensions:
            for file in glob.glob(os.path.join(img_path, ext)):
                os.remove(file)
clean_data_dirs()

# Copy and organize images
def organize_images():
    img_path = 'Data/line'
    file_list = glob.glob(os.path.join(img_path, '*.tif'))
    k1 = k2 = 0
    for file in file_list:
        if file[-16:-8] == 'Line 50x':
            dest = f"Data/train_cleaned/{k1}.tif" if k1 <= 40 else f"Data/test_cleaned/{k1}.tif"
            shutil.copyfile(file, dest)
            k1 += 1
        elif file[-16:-8] == 'Line off':
            dest = f"Data/train/{k2}.tif" if k2 <= 40 else f"Data/test/{k2}.tif"
            shutil.copyfile(file, dest)
            k2 += 1
organize_images()

def crop_image(image, xy: tuple, wh: tuple, return_grayscale=False):
    x, y = xy
    w, h = wh
    crop = image[y:y + h, x:x + w]
    if return_grayscale:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return crop

def load_and_normalize_images(img_path):
    file_list = sorted(glob.glob(os.path.join(img_path, '*.tif')))
    img_list = np.empty((len(file_list), TARGET_HEIGHT, TARGET_WIDTH, 1), dtype=np.float32)
    for i, file in enumerate(file_list):
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        image = crop_image(image, (0, 0), (512, 440))
        img = img_to_array(image).astype('float32')
        img_resized = tf.image.resize(img, size=[IMAGE_SIZE, IMAGE_SIZE], antialias=True)
        img_list[i] = tf.clip_by_value(img_resized / 255.0, 0.0, 1.0)
    return img_list

def augment_pipeline(pipeline, images, seed=5):
    ia.seed(seed)
    processed_images = images.copy()
    for step in pipeline:
        augmented = np.array(step.augment_images(images))
        processed_images = np.append(processed_images, augmented, axis=0)
    return processed_images

# Load training data
full_train = load_and_normalize_images('Data/train')
full_target = load_and_normalize_images('Data/train_cleaned')

# Define augmentation pipeline
pipeline = [
    iaa.Rot90(1), iaa.Rot90(2), iaa.Rot90(3), iaa.Rot90((1, 3)),
    iaa.PerspectiveTransform(scale=(0.02, 0.1)),
    iaa.Affine(rotate=(10)), iaa.Affine(rotate=(-10)),
    iaa.Affine(rotate=(5)), iaa.Affine(rotate=(-5)),
    iaa.Affine(rotate=(3)), iaa.Affine(rotate=(-3)),
    iaa.Affine(rotate=(7)), iaa.Affine(rotate=(-7)),
    iaa.Affine(rotate=(15)), iaa.Affine(rotate=(-15)),
    iaa.Affine(rotate=(20)), iaa.Affine(rotate=(-20)),
    iaa.Crop(px=(5, 32)), iaa.Fliplr(1), iaa.Flipud(1),
    iaa.GaussianBlur(sigma=(1, 1.5)), iaa.MotionBlur(8),
    iaa.Sequential([iaa.Rot90((1, 3)), iaa.PerspectiveTransform(scale=(0.02, 0.1))]),
    iaa.Sequential([iaa.Crop(px=(5, 32)), iaa.Fliplr(0.5), iaa.GaussianBlur(sigma=(0, 1.5))]),
    iaa.Sequential([iaa.Flipud(1), iaa.MotionBlur(k=6)])
]

# Data augmentation
processed_train = augment_pipeline(pipeline, full_train.reshape(-1, TARGET_HEIGHT, TARGET_WIDTH))
processed_target = augment_pipeline(pipeline, full_target.reshape(-1, TARGET_HEIGHT, TARGET_WIDTH))
processed_train = processed_train.reshape(-1, TARGET_HEIGHT, TARGET_WIDTH, 1)
processed_target = processed_target.reshape(-1, TARGET_HEIGHT, TARGET_WIDTH, 1)

# Build Autoencoder Model
def build_autoencoder(input_shape=(None, None, 1)):
    input_layer = Input(shape=input_shape)
    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    for _ in range(3):
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # Decoder
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    for _ in range(3):
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(input_layer, output_layer)

autoencoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='mse', metrics=['mse'])
autoencoder.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='loss', min_delta=0.001, patience=2, verbose=1)
checkpoint_loss = ModelCheckpoint('best_loss.h5', monitor='loss', save_best_only=True)
checkpoint_val_loss = ModelCheckpoint('best_val_loss.h5', monitor='val_loss', save_best_only=True)

# Train the model
history = autoencoder.fit(
    processed_train, processed_target,
    batch_size=1, epochs=40, verbose=1,
    validation_split=0.2,
    callbacks=[checkpoint_loss]
)

# Plot training history
def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['mse'], label='Train MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.title('Model MSE')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

plot_history(history)

# Save the model
autoencoder.save('AutoEncoderModelFull.h5')

# Evaluate and visualize predictions
def evaluate_and_visualize(model, x, y, indices=(11, 15)):
    preds = model.predict(x, batch_size=1)
    x, y, preds = x * 255., y * 255., preds * 255.
    i, j = indices
    fig, ax = plt.subplots(2, 3, figsize=(22, 16))
    ax[0, 0].imshow(x[i].reshape(TARGET_HEIGHT, TARGET_WIDTH), cmap='gray')
    ax[0, 1].imshow(y[i].reshape(TARGET_HEIGHT, TARGET_WIDTH), cmap='gray')
    ax[0, 2].imshow(preds[i].reshape(TARGET_HEIGHT, TARGET_WIDTH), cmap='gray')
    ax[1, 0].imshow(x[j].reshape(TARGET_HEIGHT, TARGET_WIDTH), cmap='gray')
    ax[1, 1].imshow(y[j].reshape(TARGET_HEIGHT, TARGET_WIDTH), cmap='gray')
    ax[1, 2].imshow(preds[j].reshape(TARGET_HEIGHT, TARGET_WIDTH), cmap='gray')
    plt.show()

evaluate_and_visualize(autoencoder, full_train, full_target)

# Test set evaluation
X_test = load_and_normalize_images('Data/test').reshape(-1, TARGET_HEIGHT, TARGET_WIDTH, 1)
X_test_target = load_and_normalize_images('Data/test_cleaned').reshape(-1, TARGET_HEIGHT, TARGET_WIDTH, 1)
test_preds = autoencoder.predict(X_test, batch_size=1)
X_test, test_preds = X_test * 255., test_preds * 255.

def plot_digits(*args):
    args = [x.squeeze() for x in args]
    n = min([x.shape[0] for x in args])
    plt.figure(figsize=(2 * n * 1.5, 2 * len(args) * 1.5))
    for j in range(n):
        for i in range(len(args)):
            ax = plt.subplot(len(args), n, i * n + j + 1)
            plt.imshow(args[i][j], cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()

plot_digits(X_test[:2], test_preds[:2])

def im_show(im_name):
    plt.figure(figsize=(20, 8))
    img = cv2.imread(im_name, 0)
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"[INFO] Image shape: {img.shape}")

# Evaluate model on full training set
autoencoder.evaluate(full_train, full_target, batch_size=1)
