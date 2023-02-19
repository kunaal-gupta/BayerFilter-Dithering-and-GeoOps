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
    trans_mat = None
    trans_mat_inv = None

    # TODO: implement this function (overwrite the two lines above)
    # ...
    
    return trans_mat, trans_mat_inv



def rotate_image(image):
    ''' rotate and return image '''
    h, w = image.shape[:2]
    trans_mat, trans_mat_inv = calculate_trans_mat(image)

    # TODO: determine angle and create Tr
    angle = ...
    angle_rad = ...
    Tr = np.array([])

    # TODO: compute inverse transformation to go from output to input pixel locations
    Tr_inv = ...

    out_img = np.zeros_like(image)
    for out_y in range(h):
        for out_x in range(w):
            # TODO: find input pixel location from output pixel ocation and inverse transform matrix, copy over value from input location to output location
            ...

    return out_img, Tr



def scale_image(image):
    ''' scale image and return '''
    # TODO: implement this function, similar to above
    out_img = np.zeros_like(image)
    Ts = np.array([])

    return out_img, Ts



def skew_image(image):
    ''' Skew image and return '''
    # TODO: implement this function like above
    out_img = np.zeros_like(image)
    Tskew = np.array([])

    return out_img, Tskew


def combined_warp(image):
    ''' implement your own code to perform the combined warp of rotate, scale, skew and return image + transformation matrix  '''
    # TODO: implement combined warp on your own. 
    # You need to combine the transformation matrices before performing the warp
    # (you may want to use the above functions to get the transformation matrices)
    out_img = np.zeros_like(image)
    Tc = np.array([])
    
    return out_img, Tc
    

def combined_warp_biinear(image):
    ''' perform the combined warp with bilinear interpolation (just show image) '''
    # TODO: implement combined warp -- you can use skimage.trasnform functions for this part (import if needed)
    # (you may want to use the above functions (above combined) to get the combined transformation matrix)
    out_img = np.zeros_like(image)

    return out_img



if __name__ == "__main__":
    image = read_image()
    plt.imshow(image), plt.title("Oiginal Image"), plt.show()

    rotated_img, _ = rotate_image(image)
    plt.figure(figsize=(15,5))
    plt.subplot(131),plt.imshow(rotated_img), plt.title("Rotated Image")

    scaled_img, _ = scale_image(image)
    plt.subplot(132),plt.imshow(scaled_img), plt.title("Scaled Image")

    skewed_img, _ = skew_image(image)
    plt.subplot(133),plt.imshow(skewed_img), plt.title("Skewed Image"), plt.show()

    combined_warp_img, _ = combined_warp(image)
    plt.figure(figsize=(10,5))
    plt.subplot(121),plt.imshow(combined_warp_img), plt.title("Combined Warp Image")

    combined_warp_biliear_img = combined_warp_biinear(image)
    plt.subplot(122),plt.imshow(combined_warp_biliear_img.astype(np.uint8)), plt.title("Combined Warp Image with Bilinear Interpolation"),plt.show()



