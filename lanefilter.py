import cv2
import numpy as np

#Description: The functions used to filter an image to prepare it for lane detection

def binary_thresh(array, thresh):
    """
    Make every element in threshold 1 and every element out of the threshold 0
    param array: 2D array
    param thesh: Two element tuple of min and max threshold values
    return: binary 2D thresholded data
    """

    binary = np.zeros_like(array)
    binary[(array >= thresh[0]) & (array <= thresh[1])] = 1
    return binary

def threshold(channel, thresh=(128,255), thresh_type=cv2.THRESH_BINARY):
    """
    Filter out all values outside of the threshold in the input channel
    param channel: 2D array representing one channel of an image
    param threst: Two element tuple of min and max threshold values
    param thresh_type: The opencv type of the threshold
    returns: 2D thresholded data
    """
    return cv2.threshold(channel, thresh[0], thresh[1], thresh_type)

def gaussian_blur(channel, gauss_kernel=3):
    """
    Reduce noise in the image using a gaussian blur function
    param channel: 2D array representing one channel of an image
    param gauss_kernel: The dimensions of the kernal applied
    """
    return cv2.GaussianBlur(channel, (gauss_kernel, gauss_kernel), 0)

def median_blur(channel, median_kernel=5):
    """
    Reduce noise in the image using a median blur function
    param channel: 2D array representing one channel of an image
    param median_kernel: The dimensions of the kernal applied
    """
    return cv2.medianBlur(channel, median_kernel)

def sobel(channel, sobel_kernel=3, thresh=(0, 255)):
    """
    Find edges that are aligned horizontally or vertically in the image
    param channel: 2D array representing one channel of an image
    param sobel_kernal: The dimensions (square) of the sobel kernel applied
    param thresh: Two element tuple with min and max values for binary thresholding
    return: binary 2D edge detected data
    """
    
    #detects differences in pixel intensity from left to right (veritcal edges)
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, sobel_kernel)
    #detects differences in pixel intensity from top to bottom (horizontal edges)
    sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, sobel_kernel)
    sobel_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    
    return binary_thresh(sobel_mag, thresh)

def canny(channel, canny_thresh, aperture_size=3):
    """
    Find edges using the Canny Edge Detection Algorithm
    param channel: 2D array representing one channel of an image
    param canny_thresh: two element tuple storing min and max intensity gradients values (see canny documentation)
    param aperture_size: The aperture size of the sobel kernel applied
    return: 2D edge detected data
    """
    return cv2.Canny(channel, canny_thresh[0], canny_thresh[1], aperture_size)


    









