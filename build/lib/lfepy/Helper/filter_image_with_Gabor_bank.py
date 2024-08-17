import numpy as np
from scipy.fft import fft2, ifft2
from skimage.transform import resize
from lfepy.Helper.construct_Gabor_filters import construct_Gabor_filters


def filter_image_with_Gabor_bank(image, filter_bank, down_sampling_factor=64):
    """
    Apply a Gabor filter bank to an image and return the filtered features.

    This function applies a bank of Gabor filters to an input image, performs down-sampling,
    and returns the concatenated features obtained from the filtered image. Gabor's filters are
    used for texture analysis and feature extraction in image processing.

    Args:
        image (np.ndarray): Input image to be filtered. Should be a 2D numpy array representing a grayscale image.
        filter_bank (dict): Dictionary containing Gabor filter bank with the following keys:
            'spatial': A list of 2D arrays representing spatial domain Gabor filters.
            'freq': A list of 2D arrays representing frequency domain Gabor filters.
            'orient': Number of orientations in the filter bank.
            'scales': Number of scales in the filter bank.
        down_sampling_factor (int, optional): Factor for down-sampling the filtered images. Default is 64.

    Returns:
        np.ndarray: Concatenated filtered features from the Gabor filter bank, flattened into a 1D array.

    Raises:
        ValueError: If the inputs are not as expected, dimensions do not match, or required fields are missing in the filter bank.

    Example:
        >>> import numpy as np
        >>> from skimage.data import camera
        >>> from skimage.transform import resize
        >>> from scipy.fftpack import fft2, ifft2
        >>> image = camera()
        >>> filter_bank = construct_Gabor_filters(num_of_orient=8, num_of_scales=5, size1=31)
        >>> features = filter_image_with_Gabor_bank(image, filter_bank)
        >>> print(features.shape)
    """
    # Check inputs
    if not isinstance(image, np.ndarray):
        raise ValueError("The first input parameter must be an image in the form of a numpy array.")

    if not isinstance(filter_bank, dict):
        raise ValueError("The second input parameter must be a dictionary containing the Gabor filter bank.")

    if down_sampling_factor is None:
        down_sampling_factor = 64

    if not isinstance(down_sampling_factor, (int, float)) or down_sampling_factor < 1:
        print("The down-sampling factor needs to be a numeric value larger or equal than 1! Switching to defaults: 64")
        down_sampling_factor = 64

    if 'spatial' not in filter_bank:
        raise ValueError("Could not find filters in the spatial domain. Missing filter_bank['spatial']!")

    if 'freq' not in filter_bank:
        raise ValueError("Could not find filters in the frequency domain. Missing filter_bank['freq']!")

    if 'orient' not in filter_bank:
        raise ValueError("Could not determine angular resolution. Missing filter_bank['orient']!")

    if 'scales' not in filter_bank:
        raise ValueError("Could not determine frequency resolution. Missing filter_bank['scales']!")

    # Check filter bank fields
    if not all(key in filter_bank for key in ['spatial', 'freq', 'orient', 'scales']):
        raise ValueError("Filter bank missing required fields!")

    filtered_image = []

    # Check image and filter size
    a, b = image.shape
    c, d = filter_bank['spatial'][0][0].shape

    if a == 2 * c or b == 2 * d:
        raise ValueError("The dimension of the input image and Gabor filters do not match!")

    # Compute output size
    dim_spec_down_sampl = round(np.sqrt(down_sampling_factor))
    new_size = (a // dim_spec_down_sampl, b // dim_spec_down_sampl)

    # Filter image in the frequency domain
    image_tmp = np.zeros((2 * a, 2 * b))
    image_tmp[:a, :b] = image
    image = fft2(image_tmp)

    for i in range(filter_bank['scales']):
        for j in range(filter_bank['orient']):
            # Filtering
            Imgabout = ifft2(filter_bank['freq'][i][j] * image)
            gabout = np.abs(Imgabout[a:2 * a, b:2 * b])

            # Down-sampling
            y = resize(gabout, new_size, order=1)

            # Zero mean unit variance normalization
            y = (y - np.mean(y)) / np.std(y)
            y = y.ravel()

            # Add to image
            filtered_image.append(y)

    filtered_image = np.concatenate(filtered_image)
    return filtered_image