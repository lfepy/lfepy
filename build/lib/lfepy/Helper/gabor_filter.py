import numpy as np
from lfepy.Helper.construct_Gabor_filters import construct_Gabor_filters
from lfepy.Helper.filter_image_with_Gabor_bank import filter_image_with_Gabor_bank


def gabor_filter(image, orienNum, scaleNum):
    """
    Apply a Gabor filter bank to an image and organize the results into a multidimensional array.

    Args:
        image (numpy.ndarray): Input image to be filtered. Should be a 2D numpy array.
        orienNum (int): Number of orientation filters in the Gabor filter bank.
        scaleNum (int): Number of scale filters in the Gabor filter bank.

    Returns:
        numpy.ndarray: Multidimensional array containing the Gabor magnitude responses. Shape is (height, width, orienNum, scaleNum).

    Example:
        >>> import numpy as np
        >>> from skimage.data import camera
        >>> image = camera()
        >>> gabor_magnitudes = gabor_filter(image, orienNum=8, scaleNum=5)
        >>> print(gabor_magnitudes.shape)
        (512, 512, 8, 5)
    """
    r, c = image.shape

    # Construct Gabor filter bank
    filter_bank = construct_Gabor_filters(orienNum, scaleNum, [r, c])

    # Apply Gabor filter bank to the image
    result = filter_image_with_Gabor_bank(image, filter_bank, 1)

    # Calculate number of pixels in each filter response
    pixel_num = len(result) // (orienNum * scaleNum)

    # Initialize the output array
    gaborMag = np.zeros((r, c, orienNum, scaleNum))

    # Organize the results into the output array
    orien = 0
    scale = 1
    for m in range(1, orienNum * scaleNum + 1):
        orien += 1
        if orien > orienNum:
            orien = 1
            scale += 1
        gaborMag[:, :, orien - 1, scale - 1] = result[(m - 1) * pixel_num: m * pixel_num].reshape(r, c)

    return gaborMag