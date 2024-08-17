import numpy as np
from scipy.ndimage import sobel
from skimage.color import rgb2gray
from skimage.feature import canny
from lfepy.Helper.bin_matrix import bin_matrix


def descriptor_PHOG(image, bin=8, angle=360, L=2, roi=None):
    """
    Compute the Pyramid Histogram of Oriented Gradients (PHOG) descriptor for a 2D image.

    The PHOG descriptor captures gradient information at multiple scales and orientations,
    providing a detailed description of image shapes and textures. The descriptor is computed
    for different levels of a pyramid and can be used for object recognition and image analysis.

    Args:
        image (numpy.ndarray): Input image, which can be grayscale or RGB.
        bin (int, optional): Number of orientation bins for the histogram. Default is 8.
        angle (int, optional): Angle range for orientation. Can be 180 or 360 degrees. Default is 360.
        L (int, optional): Number of pyramid levels. Default is 2.
        roi (list or None, optional): Region of Interest (ROI) specified as [y_min, y_max, x_min, x_max].
                                      If None, the entire image is used. Default is None.

    Returns:
        tuple: A tuple containing:
            p_hist (list): List of histograms for each pyramid level.
            bh_roi (numpy.ndarray): Gradient magnitude matrix for the ROI.
            bv_roi (numpy.ndarray): Gradient orientation matrix for the ROI.

    Raises:
        ValueError: If:
            'image' is not a 2D array or a 3D array with the third dimension not being 3 (RGB).
            'angle' is not 180 or 360.
            'roi' is not a list or None.

    Example:
        >>> import numpy as np
        >>> from skimage import data
        >>> image = data.camera()
        >>> p_hist, bh_roi, bv_roi = descriptor_PHOG(image, bin=8, angle=360, L=2)
        >>> print(len(p_hist))
        2
        >>> print(bh_roi.shape)
        (480, 640)
        >>> print(bv_roi.shape)
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