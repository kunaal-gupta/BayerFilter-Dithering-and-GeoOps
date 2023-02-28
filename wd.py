# Import libraries
from skimage import io
import matplotlib.pyplot as plt
import numpy as np


def read_image():
    original_img = io.imread('bird.jpeg')
    return original_img

def calculate_trans_mat(image):
    """
    return translation matrix that shifts center of image to the origin and its inverse
    """
    # TODO: implement this function (overwrite the two lines above)

    Y = int(image.shape[1]//2)
    X = int(image.shape[0]//2)

    trans_mat = np.array([[1, 0, X], [0, 1, Y], [0, 0, 1]])
    trans_mat_inv = np.array([[1, 0, -X], [0, 1, -Y], [0, 0, 1]])

    return trans_mat, trans_mat_inv



def rotate_image(image):
    ''' rotate and return image '''
    h, w = image.shape[:2]
    trans_mat, trans_mat_inv = calculate_trans_mat(image)

    # TODO: determine angle and create Tr
    angle = 75
    angle_rad = np.deg2rad(angle)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    Tr = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # TODO: compute inverse transformation to go from output to input pixel locations
    Tr_inv = np.linalg.inv(Tr)
    newimg = trans_mat @ Tr_inv @ trans_mat_inv
    out_img = np.zeros_like(image)
    for out_y in range(h):
        for out_x in range(w):
            # TODO: find input pixel location from output pixel ocation and inverse transform matrix, copy over value from input location to output location
            ...
            pixel = [out_x, out_y, 1]
            pixel = np.dot(newimg, pixel)
            if int(pixel[0]) > image.shape[0] - 1 or int(pixel[1]) > image.shape[1] - 1 or 0 > int(pixel[0]) or 0 > int(pixel[1]):
              pass
            else:
              out_img[out_x][out_y] = image[int(pixel[0])][int(pixel[1])]

    return out_img, Tr



def scale_image(image):
    ''' scale image and return '''
    # TODO: implement this function, similar to above

    h, w = image.shape[:2]
    trans_mat, trans_mat_inv = calculate_trans_mat(image)

    Ts = np.array([[2.5, 0, 0], [0, 1.5, 0], [0,0,1]])
    # TODO: compute inverse transformation to go from output to input pixel locations
    Ts_inv = np.linalg.inv(Ts)
    newimg = trans_mat @ Ts_inv @ trans_mat_inv
    out_img = np.zeros_like(image)
    for out_y in range(h):
        for out_x in range(w):
            # TODO: find input pixel location from output pixel ocation and inverse transform matrix, copy over value from input location to output location
            ...
            pixel = [out_x, out_y, 1]
            pixel = np.dot(newimg, pixel)
            if int(pixel[0]) > image.shape[0] - 1 or int(pixel[1]) > image.shape[1] - 1 or 0 > int(pixel[0]) or 0 > int(pixel[1]):
              pass
            else:
              out_img[out_x][out_y] = image[int(pixel[0])][int(pixel[1])]

    return out_img, Ts



def skew_image(image):
    ''' Skew image and return '''
    # TODO: implement this function like above
    h, w = image.shape[:2]
    trans_mat, trans_mat_inv = calculate_trans_mat(image)

    Tskew = np.array([[1, 0.2, 0], [0.2, 1, 0], [0,0,1]])
    # TODO: compute inverse transformation to go from output to input pixel locations
    Tskew_inv = np.linalg.inv(Tskew)
    newimg = trans_mat @ Tskew_inv @ trans_mat_inv
    out_img = np.zeros_like(image)
    for out_y in range(h):
        for out_x in range(w):
            # TODO: find input pixel location from output pixel ocation and inverse transform matrix, copy over value from input location to output location
            ...
            pixel = [out_x, out_y, 1]
            pixel = np.dot(newimg, pixel)
            if int(pixel[0]) > image.shape[0] - 1 or int(pixel[1]) > image.shape[1] - 1 or 0 > int(pixel[0]) or 0 > int(pixel[1]):
              pass
            else:
              out_img[out_x][out_y] = image[int(pixel[0])][int(pixel[1])]

    return out_img, Tskew

def combined_warp(image):
    ''' implement your own code to perform the combined warp of rotate, scale, skew and return image + transformation matrix  '''
    # rotate the image
    trans_mat, trans_mat_inv = calculate_trans_mat(image)
    rotated_img, Tr = rotate_image(image)
    # scale the rotated image
    scaled_img, Ts = scale_image(rotated_img)
    # skew the scaled image
    skewed_img, Tskew = skew_image(scaled_img)

    # combine the transformation matrices
    Tc = Ts @ Tr @ Tskew
    Tc2 = np.linalg.inv(Tc)
    newimg = trans_mat @ Tc2 @ trans_mat_inv

    # warp the original image using the combined transformation matrix
    h, w = image.shape[:2]
    out_img = np.zeros_like(image)
    for out_y in range(h):
        for out_x in range(w):
            # TODO: find input pixel location from output pixel ocation and inverse transform matrix, copy over value from input location to output location
            ...
            pixel = [out_x, out_y, 1]
            pixel = np.dot(newimg, pixel)
            if int(pixel[0]) > image.shape[0] - 1 or int(pixel[1]) > image.shape[1] - 1 or 0 > int(pixel[0]) or 0 > int(pixel[1]):
              pass
            else:
              out_img[out_x][out_y] = image[int(pixel[0])][int(pixel[1])]

    return out_img, Tc

from skimage.transform import warp, AffineTransform

def combined_warp_biinear(image):
    ''' perform the combined warp with bilinear interpolation (just show image) '''
    trans_mat, trans_mat_inv = calculate_trans_mat(image)
    out_img = np.zeros_like(image)
    _, Tc = combined_warp(image)
    Tc = np.linalg.inv(Tc)
    newimg = trans_mat @ Tc @ trans_mat_inv

    # apply transformation with bilinear interpolation
    out_img = warp(image, newimg, output_shape=image.shape, order=1, preserve_range=True)

    return out_img


if __name__ == "__main__":
    image = read_image()
    plt.imshow(image), plt.title("Oiginal Image"), plt.show()

    rotated_img, _ = rotate_image(image)
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(rotated_img), plt.title("Rotated Image")

    scaled_img, _ = scale_image(image)
    plt.subplot(132), plt.imshow(scaled_img), plt.title("Scaled Image")

    skewed_img, _ = skew_image(image)
    plt.subplot(133), plt.imshow(skewed_img), plt.title("Skewed Image"), plt.show()

    combined_warp_img, _ = combined_warp(image)
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(combined_warp_img), plt.title("Combined Warp Image")

    combined_warp_biliear_img = combined_warp_biinear(image)
    plt.subplot(122), plt.imshow(combined_warp_biliear_img.astype(np.uint8)), plt.title(
        "Combined Warp Image with Bilinear Interpolation"), plt.show()
