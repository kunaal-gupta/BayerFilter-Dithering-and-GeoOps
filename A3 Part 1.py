# import statements
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage import io


def part1():
    """ BasicBayer: reconstruct RGB image using GRGB pattern"""
    filename_Grayimage = 'PeppersBayerGray.bmp'
    filename_gridB = 'gridB.bmp'
    filename_gridR = 'gridR.bmp'
    filename_gridG = 'gridG.bmp'

    # read image
    img = io.imread(filename_Grayimage, as_gray=True)
    h, w = img.shape

    # our final image will be a 3 dimentional image with 3 channels
    rgb = np.zeros((h, w, 3), np.uint8)

    # reconstruction of the green channel IG
    IG = np.copy(img)  # copy the image into each channel

    for row in range(0, h, 4):  # loop step is 4 since our mask size is 4.
        for col in range(0, w, 4):  # loop step is 4 since our mask size is 4.
            # TODO: compute pixel value for each location where mask is unshaded (0)
            # interpolate each pixel using its every valid (shaded) neighbour
            IG[row, col + 1] = (int(img[row, col]) + int(img[row, col + 2]) + int(img[row + 1, col + 1])) / 3
            IG[row, col + 3] = (int(img[row, col + 2]) + int(img[row + 1, col + 3])) / 2

            IG[row + 2, col + 1] = (int(img[row + 2, col]) + int(img[row + 2, col + 2]) + int(
                img[row + 1, col + 1]) + int(img[row + 3, col + 1])) / 4
            IG[row + 2, col + 3] = (int(img[row + 2, col + 2]) + int(img[row + 1, col + 3]) + int(
                img[row + 3, col + 3])) / 3

            IG[row + 1, col] = (int(img[row, col]) + int(img[row + 2, col]) + int(img[row + 1, col + 1])) / 3
            IG[row + 1, col + 2] = (int(img[row + 1, col + 1]) + int(img[row + 1, col + 3]) + int(
                img[row, col + 1]) + int(img[row + 2, col + 1])) / 4

            IG[row + 3, col] = (int(img[row + 2, col]) + int(img[row + 3, col + 1])) / 2
            IG[row + 3, col + 2] = (int(img[row + 3, col + 1]) + int(img[row + 2, col + 2]) + int(
                img[row + 3, col + 3])) / 3

    # TODO: show green (IR) in first subplot (221) and add title - refer to rgb one for hint on plotting
    # plt.figure(figsize=(10, 8))
    # plt.imshow(IG, cmap='gray')
    # plt.show()
    # ...
    #
    #     # TODO: reconstruction of the red channel IR (simiar to loops above),
    IR = np.copy(img)  # copy the image into each channel

    for row in range(0, h, 4):  # loop step is 4 since our mask size is 4.
        for col in range(0, w, 4):  # loop step is 4 since our mask size is 4.
            # TODO: compute pixel value for each location where mask is unshaded (0)

            # First Row
            IR[row, col + 2] = (int(img[row, col + 1]) + int(img[row, col + 3])) / 2
            IR[row, col] = (int(img[row, col + 1]))

            # Second Row
            IR[row + 1, col + 1] = (int(img[row, col + 1]) + int(img[row + 2, col + 1])) / 2
            IR[row + 1, col + 2] = (int(img[row, col + 1]) + int(img[row, col + 3]) + int(img[row + 2, col + 1]) + int(
                img[row + 2, col + 3])) / 4
            IR[row + 1, col + 3] = (int(img[row, col + 3]) + int(img[row + 2, col + 3])) / 2
            IR[row + 1, col] = (int(img[row + 1, col + 1]))

            # Third Row
            IR[row + 2, col] = (int(img[row + 2, col + 1]))
            IR[row + 2, col+2] = (int(img[row + 2, col + 1]) + int(img[row + 2, col + 3])) / 2

            # Fourth Row
            IR[row + 3, col] = (int(img[row + 2, col + 1]))
            IR[row + 3, col+1] = (int(img[row + 2, col + 1]))
            IR[row + 3, col+2] = (int(img[row + 2, col + 2]))
            IR[row + 3, col+3] = (int(img[row + 2, col + 3]))

    # plt.figure(figsize=(10, 8))
    # plt.imshow(IR, cmap='gray')
    # plt.show()

    IB = np.copy(img)  # copy the image into each channel

    for row in range(0, h, 4):  # loop step is 4 since our mask size is 4.
        for col in range(0, w, 4):  # loop step is 4 since our mask size is 4.
            # TODO: compute pixel value for each location where mask is unshaded (0)

            # Fourth Row
            IB[row+3, col + 1] = (int(img[row+3, col]) + int(img[row +3 , col + 2])) / 2
            IB[row+3, col+3] = (int(img[row+3, col + 2]))

            # Third Row
            IB[row + 2, col] = (int(img[row+1, col]) + int(img[row + 3, col])) / 2
            IB[row + 2, col + 1] = (int(img[row+1, col]) + int(img[row+1, col + 2]) + int(img[row + 3, col]) + int(
                img[row + 3, col + 2])) / 4
            IB[row + 2, col + 2] = (int(img[row + 1, col + 2]) + int(img[row + 3, col + 2])) / 2
            IB[row + 2, col + 3] = (int(img[row + 2, col + 2]))

            # Second Row
            IB[row + 1, col+1] = (int(img[row + 1, col]) + int(img[row + 1, col+2]))/2
            IB[row + 1, col+3] = (int(img[row + 1, col + 2]))

            # First Row
            IB[row, col] = (int(img[row + 1, col]))
            IB[row, col+1] = (int(img[row + 1, col + 1]))
            IB[row, col+2] = (int(img[row + 1, col + 2]))
            IB[row, col+3] = (int(img[row + 1, col + 3]))


    # plt.figure(figsize=(10, 8))
    # plt.imshow(IB, cmap='gray')
    # plt.show()

    rgb = np.dstack((IR, IG, IB))
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb, cmap='gray')
    plt.show()



#     #
#     # TODO: show IR in second subplot (224) and title
#     # ...
#
#     # TODO: reconstruction of the blue channel IB (similar to loops above),
#     # ...
#     # TODO: show IB in third subplot () and title
#     # ...
#
#     # TODO: merge the three channels IG, IB, IR in the correct order
#     rgb[:, :, 1] = IG
#     # ...
#
#     # TODO: show rgb image in final subplot (224) and add title
#     plt.subplot(224)
#     plt.imshow(rgb), plt.title('rgb')
#     plt.show()
#
#
# if __name__ == "__main__":
part1()
