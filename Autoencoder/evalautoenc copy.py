"""
Autoencoder Evaluation Script

Author: Seoleun Shin (KRISS), 2023-June

This script evaluates a denoising autoencoder using various image similarity metrics.
Metrics are based on https://github.com/up42/image-similarity-measures and adapted for SEM images.

"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from image_similarity_measures.evaluate import evaluation
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(456)

# Constants
TARGET_HEIGHT = 512
TARGET_WIDTH = 512
IMAGE_SIZE = 512
NOISY_DIR = './Data/Noisy'
CLEAN_DIR = './Data/Clean'
MODEL_PATH = 'AutoEncoderModelFull.h5'
RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def crop_image(image, xy: tuple, wh: tuple, return_grayscale=False):
    x, y = xy
    w, h = wh
    crop = image[y:y + h, x:x + w]
    if return_grayscale:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return crop

def load_and_preprocess_images(img_dir):
    file_list = sorted(glob.glob(os.path.join(img_dir, '*.tif')))
    img_list = np.empty((len(file_list), TARGET_HEIGHT, TARGET_WIDTH, 1), dtype=np.float32)
    for i, file_path in enumerate(file_list):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = crop_image(img, (0, 0), (512, 440))
        img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
        img = img.astype('float32') / 255.0
        img_list[i, ..., 0] = img
    return img_list

def build_autoencoder():
    input_layer = Input(shape=(None, None, 1))
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

def evaluate_model(model, noisy_images, clean_images):
    psnr_list, ssim_list, rmse_list, fsim_list, uiq_list = [], [], [], [], []
    for idx, (noisy, clean) in enumerate(zip(noisy_images, clean_images), 1):
        noisy_input = noisy.reshape(1, TARGET_HEIGHT, TARGET_WIDTH, 1)
        clean_input = clean.reshape(1, TARGET_HEIGHT, TARGET_WIDTH, 1)
        denoised = model.predict(noisy_input, batch_size=1)
        # Scale back to [0,255] for metrics
        noisy_img = (noisy * 255).astype(np.uint8)
        clean_img = (clean * 255).astype(np.uint8)
        denoised_img = (denoised.squeeze() * 255).astype(np.uint8)
        # Metrics
        psnr = peak_signal_noise_ratio(clean_img, denoised_img)
        ssim = structural_similarity(clean_img, denoised_img, data_range=255, multichannel=False)
        rmse = evaluation(clean_img, denoised_img, metrics=["rmse"])
        fsim = evaluation(clean_img, denoised_img, metrics=["fsim"])
        uiq = evaluation(clean_img, denoised_img, metrics=["uiq"])
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        rmse_list.append(rmse)
        fsim_list.append(fsim)
        uiq_list.append(uiq)
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].imshow(noisy_img, cmap='gray')
        axes[0].set_title("Noisy Image")
        axes[1].imshow(clean_img, cmap='gray')
        axes[1].set_title("Clean Image")
        axes[2].imshow(denoised_img, cmap='gray')
        axes[2].set_title("Denoised Image")
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'eval_result_{idx}.png'))
        plt.close(fig)
    return psnr_list, ssim_list, rmse_list, fsim_list, uiq_list

def print_metrics(psnr, ssim, rmse, fsim, uiq):
    print(f"PSNR:  mean={np.mean(psnr):.4f}, std={np.std(psnr):.4f}")
    print(f"SSIM:  mean={np.mean(ssim):.4f}, std={np.std(ssim):.4f}")
    print(f"FSIM:  mean={np.mean(fsim):.4f}, std={np.std(fsim):.4f}")
    print(f"UIQ:   mean={np.mean(uiq):.4f}, std={np.std(uiq):.4f}")
    print(f"RMSE:  mean={np.mean(rmse):.4f}, std={np.std(rmse):.4f}")

def main():
    # Load images
    noisy_images = load_and_preprocess_images(NOISY_DIR)
    clean_images = load_and_preprocess_images(CLEAN_DIR)
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"Model file '{MODEL_PATH}' not found.")
        return
    model = load_model(MODEL_PATH)
    # Evaluate
    psnr, ssim, rmse, fsim, uiq = evaluate_model(model, noisy_images, clean_images)
    print_metrics(psnr, ssim, rmse, fsim, uiq)

if __name__ == "__main__":
    main()
