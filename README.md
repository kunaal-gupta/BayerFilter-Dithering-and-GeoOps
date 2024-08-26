# Image Processing Scripts

This repository contains various scripts for image processing, including basic Bayer pattern reconstruction, color dithering using Floyd-Steinberg algorithm, and image transformations such as rotation, scaling, and skewing. The code uses Python libraries like NumPy, Matplotlib, scikit-image, and SciPy.

## Table of Contents

1. [Basic Bayer Pattern Reconstruction](#basic-bayer-pattern-reconstruction)
2. [Color Dithering](#color-dithering)
3. [Image Transformations](#image-transformations)

## Basic Bayer Pattern Reconstruction

The `part1` function performs reconstruction of an RGB image from a grayscale Bayer pattern image using the GRGB pattern. The process includes:
- Interpolation of the green channel (`IG`)
- Reconstruction of the red channel (`IR`)
- Reconstruction of the blue channel (`IB`)

## Usage

1. Place the input image files (`PeppersBayerGray.bmp`, `gridB.bmp`, `gridR.bmp`, `gridG.bmp`) in the working directory.
2. Run the script to display the reconstructed RGB image and its channels.

```bash
python script_name.py
```

## Color Dithering
This section includes functions for dithering an image using Floyd-Steinberg dithering algorithm and clustering colors with KMeans. The primary functions are:

- findPalette: Generates a color palette using KMeans clustering.
- ModifiedFloydSteinbergDitherColor: Applies Floyd-Steinberg dithering using the generated palette.

## Image Transformations
The script includes functions for performing image transformations:

- rotate_image: Rotates the image by a specified angle.
- scale_image: Scales the image by a specified factor.
- skew_image: Skews the image by a specified factor.
- combined_warp: Applies a combination of rotation, scaling, and skewing transformations.
- combined_warp_bilinear: Performs combined warp with bilinear interpolation.

## Requirements
Ensure the following libraries are installed:

- NumPy
- Matplotlib
- scikit-image
- SciPy
- scikit-learn

```bash
pip install numpy matplotlib scikit-image scipy scikit-learn
```
