# Import libraries
from skimage import io
from skimage import exposure
import skimage
from skimage.color import rgb2gray
from skimage import transform
import matplotlib.pyplot as plt
import numpy as np


def part4():
    ''' Stitch two images together '''
    image0 = io.imread('im1.jpg', True)
    image1 = io.imread('im2.jpg', True)


    plt.figure(figsize=(8,12))
    plt.subplot(221),plt.imshow(image0,cmap='gray'),plt.title("First Image")
    plt.subplot(222),plt.imshow(image1,cmap='gray'),plt.title("Second Image")


    # -------- Feature detection and matching -----

    # TODO: Initiate ORB detector
    from skimage.feature import ORB, match_descriptors
    # ...

    # TODO: Find the keypoints and descriptors
    # ...

    # TODO: initialize Brute-Force matcher and exclude outliers. See match descriptor function.
    # ...

    # -------- Transform estimation -------

    # TODO: Compute homography matrix using ransac and ProjectiveTransform
    from skimage.measure import ransac
    from skimage.transform import ProjectiveTransform
    # ...
    # model_robust, inliers = ransac ...

    # ------------- Warping ----------------
    #Next, we produce the panorama itself. The first step is to find the shape of the output image by considering the extents of all warped images.

    r, c = image1.shape[:2]

    # Note that transformations take coordinates in
    # (x, y) format, not (row, column), in order to be
    # consistent with most literature.
    corners = np.array([[0, 0],
                        [0, r],
                        [c, 0],
                        [c, r]])

    # Warp the image corners to their new positions.
    warped_corners = model_robust(corners)

    # Find the extents of both the reference image and
    # the warped target image.
    all_corners = np.vstack((warped_corners, corners))

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)

    output_shape = (corner_max - corner_min)
    output_shape = np.ceil(output_shape[::-1])

    # ----- Note: The images are now warped according to the estimated transformation model.

    # A shift is added to ensure that both images are visible in their entirety. Note that warp takes the inverse mapping as input.
    from skimage.color import gray2rgb
    from skimage.exposure import rescale_intensity
    from skimage.transform import warp
    from skimage.transform import SimilarityTransform

    offset = SimilarityTransform(translation=-corner_min)

    image0_ = warp(image0, offset.inverse,
                output_shape=output_shape)

    image1_ = warp(image1, (model_robust + offset).inverse,
                output_shape=output_shape)

    plt.subplot(223),plt.imshow(image0_, cmap="gray"),plt.title("Warped first image")
    plt.subplot(224),plt.imshow(image1_, cmap="gray"),plt.title("Warped first image")
    plt.show()

    #An alpha channel is added to the warped images before merging them into a single image:

    def add_alpha(image, background=-1):
        """Add an alpha layer to the image.

        The alpha layer is set to 1 for foreground
        and 0 for background.
        """
        rgb = gray2rgb(image)
        alpha = (image != background)
        return np.dstack((rgb, alpha))





    # TODO: add alpha to the image0 and image1
    # ...

    # TODO: merge the alpha added image (only change the next line)
    # merged = ...
    alpha = merged[..., 3]
    merged /= np.maximum(alpha, 1)[..., np.newaxis]

    # The summed alpha layers give us an indication of
    # how many images were combined to make up each
    # pixel.  Divide by the number of images to get
    # an average.


    plt.figure(figsize=(10,8))
    plt.imshow(merged, cmap="gray")
    plt.show()


if __name__ == "__main__":
    part4()