import cv2
import numpy as np
import sys
import math

def convolve(image, mask):
    width = image.shape[1]
    height = image.shape[0]
    w_range = int(math.floor(mask.shape[0]/2))

    res_image = np.zeros((height, width))
    # Iterate over every pixel that can be covered by the mask
    for i in range(w_range,width-w_range):
        for j in range(w_range,height-w_range):
            # Then convolute with the mask 
            for k in range(-1*w_range,w_range):
                for h in range(-1*w_range,w_range):
                    res_image[j, i] += mask[w_range+h,w_range+k]*image[j+h,i+k]
    return res_image

if len(sys.argv) > 1:
    np.seterr(divide = 'ignore')
    
    user_input = sys.argv[1:]
    image = cv2.imread(user_input[0], cv2.IMREAD_GRAYSCALE)

    # Define the LoG filter parameters
    sigma = 2
    k_size = int(np.ceil(6 * sigma)) + 1

    kernel_size = int(2*np.ceil(3*sigma)+1)
    gauss_kernel = np.zeros((kernel_size,kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - kernel_size//2, j - kernel_size//2
            gauss_kernel[i,j] = (x**2 + y**2 - 2*sigma**2) / sigma**4 * np.exp(-(x**2+y**2)/(2*sigma**2))
    gauss_kernel /= gauss_kernel.sum()

    # Create a Laplacian kernel
    laplace_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # Convolve the image with the Gaussian kernel
    smoothed = convolve(image, gauss_kernel)

    # Convolve the smoothed image with the Laplacian kernel
    laplacian_image = convolve(smoothed, laplace_kernel)

    # Convolve the Laplacian image with the Gaussian kernel to compute the LoG response
    log_image = convolve(laplacian_image, gauss_kernel * sigma**2)

    # Normalize the output image
    log_image = (log_image - np.min(log_image)) / (np.max(log_image) - np.min(log_image)) * 255

    # Saving the result in local system
    cv2.imwrite("LoG_output.jpg", log_image)

    print("LoG image has been created")
else:
    print("No command line input provided.")

