import numpy as np
from scipy import signal, ndimage
import cv2

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:        (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # Compute image gradients
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    Gx = np.array([[0.0, -0.5, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0]])
    Gy = Gx.transpose()
    Ix = signal.convolve2d(img, Gx, mode='same')
    Iy = signal.convolve2d(img, Gy, mode='same')


    # Compute local auto-correlation matrix
    # TODO: compute the auto-correlation matrix here
    # You may refer to cv2.GaussianBlur for the gaussian filtering (borderType=cv2.BORDER_REPLICATE)
    Ix2 = Ix * Ix
    Ixy = Ix * Iy
    Iy2 = Iy * Iy
    wIx2 = cv2.GaussianBlur(Ix2, (3,3), sigma, sigma, borderType=cv2.BORDER_REPLICATE)
    wIxy = cv2.GaussianBlur(Ixy, (3,3), sigma, sigma, borderType=cv2.BORDER_REPLICATE)
    wIy2 = cv2.GaussianBlur(Iy2, (3,3), sigma, sigma, borderType=cv2.BORDER_REPLICATE)


    # Compute Harris response function
    # TODO: compute the Harris response function C here
    C = wIx2 * wIy2 - wIxy ** 2  - k * (wIx2 + wIy2) ** 2
    # cv2.imwrite('test.jpg', (C>thresh) * 255)


    # Detection with threshold
    # TODO: detection and find the corners here
    # For the local maximum check, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    C_max3 = ndimage.maximum_filter(C, size=3)
    corners = np.nonzero((C > thresh) * (np.abs(C-C_max3)<1e-17))
    corners = np.array([corners[1], corners[0]]).transpose()
    return corners, C

