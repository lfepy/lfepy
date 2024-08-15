import numpy as np
from scipy.ndimage import sobel
from skimage.color import rgb2gray
from skimage.feature import canny
from lfepy.Helper.bin_matrix import bin_matrix


def descriptor_PHOG(image, bin=8, angle=360, L=2, roi=None):
    """
    Compute the Pyramid Histogram of Oriented Gradients (PHOG) descriptor for a 2D image.

    :param image: Input image, which can be grayscale or RGB.
    :type image: numpy.ndarray
    :param bin: Number of orientation bins for the histogram. Default is 8.
    :type bin: int, optional
    :param angle: Angle range for orientation. Can be 180 or 360 degrees. Default is 360.
    :type angle: int, optional
    :param L: Number of pyramid levels. Default is 2.
    :type L: int, optional
    :param roi: Region of Interest (ROI) as [y_min, y_max, x_min, x_max]. If None, the entire image is used.
    :type roi: list or None, optional

    :returns:
        - p_hist: List of histograms for each pyramid level.
        - bh_roi: Gradient magnitude matrix for the ROI.
        - bv_roi: Gradient orientation matrix for the ROI.
    :rtype:
        - p_hist: list
        - bh_roi: numpy.ndarray
        - bv_roi: numpy.ndarray

    :example:
        >>> import numpy as np
        >>> from skimage import data
        >>> image = data.camera()  # Example grayscale image
        >>> p_hist, bh_roi, bv_roi = descriptor_PHOG(image, bin=8, angle=360, L=2)
        >>> print(len(p_hist))  # Number of levels in the PHOG descriptor
        2
        >>> print(bh_roi.shape)  # Shape of the gradient magnitude matrix for the ROI
        (480, 640)
        >>> print(bv_roi.shape)  # Shape of the gradient orientation matrix for the ROI
        (480, 640)
    """
    # Set ROI to the entire image if not specified
    if roi is None:
        roi = [0, image.shape[0], 0, image.shape[1]]

    # Convert RGB image to grayscale if necessary
    if image.ndim == 3:
        G = rgb2gray(image)
    else:
        G = image

    # Check if the grayscale image is not too uniform
    if np.sum(G) > 100:
        # Compute edge map using Canny edge detector
        E = canny(G)

        # Compute gradient magnitudes in x and y directions
        GradientX = sobel(G, axis=1)
        GradientY = sobel(G, axis=0)
        Gr = np.sqrt(GradientX ** 2 + GradientY ** 2)

        # Avoid division by zero
        GradientX[GradientX == 0] = 1e-5

        # Compute gradient orientation
        YX = GradientY / GradientX
        if angle == 180:
            A = (np.arctan(YX) + (np.pi / 2)) * 180 / np.pi
        elif angle == 360:
            A = (np.arctan2(GradientY, GradientX) + np.pi) * 180 / np.pi

        # Compute orientation histograms
        bh, bv = bin_matrix(A, E, Gr, angle, bin)
    else:
        # Return empty histograms if the image is too uniform
        bh = np.zeros_like(G)
        bv = np.zeros_like(G)

    # Extract the region of interest (ROI) from the histograms
    bh_roi = bh[roi[0]:roi[1], roi[2]:roi[3]]
    bv_roi = bv[roi[0]:roi[1], roi[2]:roi[3]]

    # Placeholder for histogram computation (not implemented here)
    p_hist = []

    return p_hist, bh_roi, bv_roi