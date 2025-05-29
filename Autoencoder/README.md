# Autoencoder

This project provides scripts for training and evaluating a convolutional autoencoder for SEM image denoising.

## Scripts

- **`semautoenc.py`**: Training script.
- **`evalautoenc.py`**: Validation and evaluation script.

---

## Functions

- **`cropImage(Image, XY, WH, returnGrayscale=False)`**  
    Crops a region from an image using top-left coordinates (`XY`) and width/height (`WH`). Optionally converts the cropped image to grayscale.

- **`renormal_load_image_from_dir2(img_path)`**  
    Loads all `.tif` images from a directory. Crops, resizes, and normalizes them to shape `(512, 512, 1)`.

- **`train_test_split(data, random_seed=55, split=0.75)`**  
    Splits data into training and validation sets using a random seed.

- **`augment_pipeline(pipeline, images, seed=5)`**  
    Applies a sequence of augmentations to images using `imgaug`.

- **`my_loss(y_true, y_pred)`**  
    Custom loss function combining MSE and SSIM.

- **`ssim_loss(y_true, y_pred)`**  
    Loss based on SSIM.

- **`msssim_loss(y_true, y_pred)`**  
    Loss based on multi-scale SSIM.

---

## Main Steps

1. **Imports and Setup**  
     - Imports TensorFlow, Keras, image processing, and evaluation libraries.
     - Sets up GPU/CPU configuration.

2. **Model Definition**  
     - Defines a convolutional autoencoder architecture using Keras functional API.

3. **Model Compilation**  
     - Compiles the model with Adam optimizer and MSE metric.

4. **Callbacks**  
     - Sets up early stopping and model checkpointing.

5. **Model Loading**  
     - Loads a pre-trained autoencoder model from file (`AutoEncoderModelFull.h5`).

6. **Data Loading**  
     - Loads noisy and clean images from `./Data/Noisy` and `./Data/Clean` directories.

7. **Evaluation Loop**  
     - For each pair of noisy/clean images:
         - Predicts denoised output using the autoencoder.
         - Calculates metrics: PSNR, SSIM, RMSE, FSIM, UIQ.
         - Plots and saves comparison images.

8. **Results Aggregation**  
     - Computes and prints average and standard deviation for each metric.

---

## Summary

The scripts load a pre-trained autoencoder, evaluate it on noisy/clean image pairs, compute several image quality metrics (PSNR, SSIM, RMSE, FSIM, UIQ), save visual results, and print summary statistics. Functions are provided for image cropping, loading, augmentation, and custom loss calculations.

---
