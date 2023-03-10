import os
from sklearn.cluster import KMeans
from scipy import spatial
from skimage import io, color, img_as_float
import numpy as np
import matplotlib.pyplot as plt
from math import floor


# Finds the closest colour in the palette using kd-tree.
def nearest(palette, colour):
    dist, i = palette.query(colour)
    return palette.data[i]


# Make a kd-tree palette from the provided list of colours
def makePalette(colours):
    return spatial.KDTree(colours)


# Dynamically calculates an N-colour palette for the given image
# Uses the KMeans clustering algorithm to determine the best colours
# Returns a kd-tree palette with those colours
def findPalette(image, nColours):
    # TODO: perform KMeans clustering to get 'Colors' --  the computed k means
    w = tuple(image.shape)[0]
    h = tuple(image.shape)[1]
    d = tuple(image.shape)[2]
    image_array = np.reshape(image, (w * h, d))

    k_means = KMeans(n_clusters=nColours, random_state=0).fit(image_array)
    Colors = k_means.cluster_centers_

    colours_img = np.zeros((50, int(nColours*50), 3), dtype=np.float32)
    start_id = 0

    for col_id in range(1, nColours):
        end_id = start_id + 50
        colours_img[:, start_id:end_id, :] = Colors[col_id, :]
        start_id = end_id

    print(f'Colors:\n{Colors}')

    plt.figure(figsize=(10, 5))
    plt.imshow(colours_img)

    return makePalette(Colors)


def ModifiedFloydSteinbergDitherColor(image, palette):

    # TODO: implement agorithm for RGB image (hint: you need to handle error in each channel separately)
    total_abs_err = 0

    for y in np.arange(0, image.shape[0]-1, 1):
        for x in np.arange(0, image.shape[1]-1, 1):
            OldPixel = image[x][y].copy()
            NewPixel = nearest(palette, OldPixel) # Determine the new colour for the current pixel from palette

            image[x][y] = NewPixel
            quant_error = OldPixel - NewPixel

            total_abs_err = total_abs_err + abs(quant_error)

            image[x + 1][y] = image[x + 1][y] + quant_error * 11 / 26
            image[x - 1][y + 1] = image[x - 1][y + 1] + quant_error * 5 / 26
            image[x][y + 1] = image[x][y + 1] + quant_error * 7 / 26
            image[x + 1][y + 1] = image[x + 1][y + 1] + quant_error * 3 / 26

    return image


if __name__ == "__main__":
    # The number colours: change to generate a dynamic palette
    nColours = 7

    # read image
    imfile = 'mandrill.png'
    image = io.imread(imfile)
    orig = image.copy()

    # Strip the alpha channel if it exists
    image = image[:, :, :3]

    # Convert the image from 8bits per channel to floats in each channel for precision
    image = img_as_float(image)

    # Dynamically generate an N colour palette for the given image
    palette = findPalette(image, nColours)
    colours = palette.data
    colours = img_as_float([colours.astype(np.ubyte)])[0]

    # call dithering function
    img = ModifiedFloydSteinbergDitherColor(image, palette)

    # show
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(orig), plt.title('Original Image')
    plt.subplot(122), plt.imshow(img), plt.title(f'Dithered Image (nColours = {nColours})')
    plt.show()
