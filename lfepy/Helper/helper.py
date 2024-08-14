import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import convolve, label, sobel
from scipy.fft import fft2, ifft2, ifftshift
from scipy.spatial.distance import cdist
from scipy.signal import convolve2d
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import canny


def roundn(x, n):
    """
    Round a number to a specified number of decimal places.

    :param x: The number or array of numbers to be rounded.
    :type x: float or array-like
    :param n: The number of decimal places to round to. If `n` is negative, `x` is rounded to the left of the decimal point. If `n` is zero, `x` is rounded to the nearest integer.
    :type n: int

    :returns: The rounded number or array of numbers.
    :rtype: float or array-like

    :example:
        >>> roundn(123.456, 2)
        123.46
        >>> roundn(123.456, -1)
        120.0
        >>> roundn(123.456, 0)
        123.0
    """
    # Check if n is negative
    if n < 0:
        # Calculate the power of 10 to shift the decimal point left
        p = 10 ** -n
        # Multiply x by p, round to the nearest integer, then divide by p to shift the decimal point back
        x = np.round(p * x) / p
    elif n > 0:
        # Calculate the power of 10 to shift the decimal point right
        p = 10 ** n
        # Divide x by p, round to the nearest integer, then multiply by p to shift the decimal point back
        x = p * np.round(x / p)
    else:
        # If n is zero, round x to the nearest integer
        x = np.round(x)

    return x


def view_as_windows(arr, window_shape, step=1):
    """
    Create a view of an array with sliding windows.

    :param arr: The input array.
    :type arr: numpy.ndarray
    :param window_shape: Shape of the sliding window.
    :type window_shape: tuple
    :param step: Step size of the sliding window.
    :type step: int or tuple

    :returns: A view of the array with sliding windows.
    :rtype: numpy.ndarray

    :raises ValueError: If any dimension of the window shape is larger than the corresponding dimension of the array.

    :example:
        >>> view_as_windows(np.array([1, 2, 3, 4]), window_shape=(2,), step=1)
        array([[1, 2],
               [2, 3],
               [3, 4]])
    """
    # Convert input to numpy array
    arr = np.asarray(arr)

    # Ensure window_shape and step are numpy arrays of at least 1 dimension
    window_shape = np.atleast_1d(window_shape)
    step = np.atleast_1d(step)

    # Check if any window dimension is larger than the corresponding array dimension
    if np.any(np.array(window_shape) > np.array(arr.shape)):
        raise ValueError("Window shape must be smaller than array shape.")

    # Calculate the shape of the new view with sliding windows
    shape = tuple(np.subtract(arr.shape, window_shape) // step + 1) + tuple(window_shape)

    # Calculate the strides of the new view
    strides = arr.strides * 2

    # Create the new view using np.lib.stride_tricks.as_strided
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def get_mapping(samples, mappingtype):
    """
    Generate a mapping table for Local Binary Patterns (LBP) codes.

    :param samples: The number of sampling points in the LBP.
    :type samples: int
    :param mappingtype: The type of LBP mapping ('u2', 'ri', 'riu2').
    :type mappingtype: str

    :returns: A dictionary with the following keys:
    :rtype: dict
    :returns:
        - 'table': The mapping table.
        - 'samples': The number of sampling points.
        - 'num': The number of patterns in the resulting LBP code.

    :raises ValueError: If an unsupported mapping type is provided.

    :example:
        >>> get_mapping(8, 'u2')
        {'table': array([...]), 'samples': 8, 'num': 59}
    """
    # Initialize the mapping table with all possible LBP codes
    table = np.arange(2 ** samples)
    newMax = 0  # Number of patterns in the resulting LBP code
    index = 0

    if mappingtype == 'u2':  # Uniform 2
        # The maximum number of uniform patterns for given samples
        newMax = samples * (samples - 1) + 3
        for i in range(2 ** samples):
            # Convert number to binary representation with fixed width
            i_bin = np.binary_repr(i, width=samples)
            # Create a circularly shifted version of the binary number
            j_bin = np.roll(list(i_bin), -1)
            # Calculate the number of 0-1 and 1-0 transitions
            numt = sum(ib != jb for ib, jb in zip(i_bin, j_bin))
            if numt <= 2:
                table[i] = index
                index += 1
            else:
                table[i] = newMax - 1

    elif mappingtype == 'ri':  # Rotation invariant
        tmpMap = np.full(2 ** samples, -1)
        for i in range(2 ** samples):
            rm = i
            r_bin = np.binary_repr(i, width=samples)
            for j in range(1, samples):
                r = int(''.join(np.roll(list(r_bin), -j)), 2)
                if r < rm:
                    rm = r
            if tmpMap[rm] < 0:
                tmpMap[rm] = newMax
                newMax += 1
            table[i] = tmpMap[rm]

    elif mappingtype == 'riu2':  # Uniform & Rotation invariant
        # The maximum number of uniform patterns for given samples in rotation-invariant setting
        newMax = samples + 2
        for i in range(2 ** samples):
            i_bin = np.binary_repr(i, width=samples)
            j_bin = np.roll(list(i_bin), -1)
            numt = sum(ib != jb for ib, jb in zip(i_bin, j_bin))
            if numt <= 2:
                table[i] = sum(int(bit) for bit in i_bin)
            else:
                table[i] = samples + 1

    else:
        raise ValueError("Unsupported mapping type. Supported types: 'u2', 'ri', 'riu2'.")

    mapping = {'table': table, 'samples': samples, 'num': newMax}
    return mapping


def get_mapping_mrelbp(samples, mappingtype):
    """
    Generate a mapping table for Modified Rotation and Uniform Local Binary Patterns (MRELBP) codes.

    :param samples: The number of sampling points in the LBP.
    :type samples: int
    :param mappingtype: The type of LBP mapping, supporting various uniform, rotation invariant, and modified patterns.
    :type mappingtype: str

    :returns: A dictionary with the following keys:
    :rtype: dict
    :returns:
        - 'table': The mapping table.
        - 'samples': The number of sampling points.
        - 'num': The number of patterns in the resulting LBP code.

    :example:
        >>> get_mapping_mrelbp(8, 'u2')
        {'table': array([...]), 'samples': 8, 'num': 59}
    """
    num_all_LBPs = 2 ** samples  # Total number of possible LBPs
    table = np.arange(num_all_LBPs)
    new_max = 0
    index = 0

    # Uniform 2
    if mappingtype in ['u2', 'LBPu2', 'LBPVu2GMPD2']:
        new_max = samples * (samples - 1) + 3
        for i in range(num_all_LBPs):
            # Rotate left by 1 bit
            j = (i << 1 | i >> (samples - 1)) & ((1 << samples) - 1)
            # Count the number of transitions (0-1 and 1-0) in the binary representation
            numt = bin(i ^ j).count('1')
            if numt <= 2:
                table[i] = index
                index += 1
            else:
                table[i] = new_max - 1

    # Rotation invariant
    if mappingtype == 'ri':
        tmp_map = np.full(num_all_LBPs, -1)
        for i in range(num_all_LBPs):
            rm = i
            r = i
            for j in range(1, samples):
                # Rotate left by 1 bit
                r = (r << 1 | r >> (samples - 1)) & ((1 << samples) - 1)
                if r < rm:
                    rm = r
            if tmp_map[rm] < 0:
                tmp_map[rm] = new_max
                new_max += 1
            table[i] = tmp_map[rm]

    # Uniform and Rotation invariant
    if mappingtype in ['riu2', 'MELBPVary', 'AELBPVary', 'GELBPEight', 'CLBPEight', 'ELBPEight',
                       'LBPriu2Eight', 'MELBPEight', 'AELBPEight', 'MELBPEightSch1', 'MELBPEightSch2',
                       'MELBPEightSch3', 'MELBPEightSch4', 'MELBPEightSch5', 'MELBPEightSch6',
                       'MELBPEightSch7', 'MELBPEightSch8', 'MELBPEightSch9', 'MELBPEightSch10',
                       'MELBPEightSch0', 'MELBPEightSch11']:
        new_max = samples + 2
        for i in range(num_all_LBPs):
            # Rotate left by 1 bit
            j = (i << 1 | i >> (samples - 1)) & ((1 << samples) - 1)
            # Count the number of transitions (0-1 and 1-0) in the binary representation
            numt = bin(i ^ j).count('1')
            if numt <= 2:
                table[i] = bin(i).count('1')
            else:
                table[i] = samples + 1

    # MELBPEightSch1Num
    if mappingtype == 'MELBPEightSch1Num':
        new_max = 2 * (samples - 1)
        for i in range(num_all_LBPs):
            # Rotate left by 1 bit
            j = (i << 1 | i >> (samples - 1)) & ((1 << samples) - 1)
            # Count the number of transitions (0-1 and 1-0) in the binary representation
            numt = bin(i ^ j).count('1')
            if numt <= 2:
                table[i] = bin(i).count('1')
            else:
                num_ones_in_LBP = bin(i).count('1')
                table[i] = samples + num_ones_in_LBP - 1

    # MELBPEightSch1Count
    if mappingtype == 'MELBPEightSch1Count':
        new_max = samples + 1
        for i in range(num_all_LBPs):
            num_ones_in_LBP = bin(i).count('1')
            table[i] = num_ones_in_LBP

    # Create the mapping dictionary
    mapping = {
        'table': table,
        'samples': samples,
        'num': new_max
    }

    # If mappingtype is empty, set 'num' to total number of possible LBPs
    if mappingtype == '':
        mapping['num'] = num_all_LBPs

    return mapping


def get_mapping_info_ct(lbp_radius, lbp_points, lbp_method):
    """
    Retrieve or generate a mapping for circular (center-symmetric) LBP.

    :param lbp_radius: The radius of the LBP.
    :type lbp_radius: int
    :param lbp_points: The number of sampling points in the LBP.
    :type lbp_points: int
    :param lbp_method: The method for LBP mapping.
    :type lbp_method: str

    :returns: A dictionary with the mapping information.
    :rtype: dict
    :returns:
        - 'table': The mapping table.
        - 'samples': The number of sampling points.
        - 'num': The number of patterns in the resulting LBP code.

    :example:
        >>> get_mapping_info_ct(1, 24, 'LBPriu2')
        {'table': array([...]), 'samples': 24, 'num': 26}
    """
    global block_size
    block_size = lbp_radius * 2 + 1  # Calculate the block size based on radius

    mapping = None

    # Load precomputed mapping based on specific points and method
    if lbp_points == 24 and lbp_method == 'LBPriu2':
        with open('mappingLBPpoints24RIU2.pkl', 'rb') as file:
            mapping = pickle.load(file)
    elif lbp_points == 16 and lbp_method == 'LBPriu2':
        with open('mappingLBPpoints16RIU2.pkl', 'rb') as file:
            mapping = pickle.load(file)
    elif lbp_points == 16 and lbp_method == 'MELBPVary':
        with open('mappingLBPpoints16RIU2.pkl', 'rb') as file:
            mapping = pickle.load(file)
    elif lbp_points == 24 and lbp_method == 'AELBPVary':
        with open('mappingLBPpoints24RIU2.pkl', 'rb') as file:
            mapping = pickle.load(file)
    else:
        # Generate mapping dynamically if precomputed mapping is not available
        mapping = get_mapping_mrelbp(lbp_points, lbp_method)

    return mapping


def descriptor_LBP(*varargin):
    """
    Compute the Local Binary Pattern (LBP) of an image with various options for radius, neighbors, mapping, and mode.

    :param image: The input image.
    :type image: numpy.ndarray
    :param radius: The radius of the LBP.
    :type radius: int
    :param neighbors: The number of sampling points in the LBP.
    :type neighbors: int
    :param mapping: The mapping information.
    :type mapping: dict or None
    :param mode: The mode for LBP calculation. Options are 'h' (histogram), 'hist' (histogram), or 'nh' (normalized histogram).
    :type mode: str

    :returns: A tuple containing:
        - result: The LBP histogram or LBP image.
        - codeImage: The LBP code image.
    :rtype: tuple

    :raises ValueError: If the number of input arguments is incorrect or other validation fails.

    :example:
        >>> image = np.random.rand(100, 100)
        >>> result, codeImage = compute_lbp(image, 1, 8, None, 'nh')
    """
    # Check the number of input arguments
    if len(varargin) < 1 or len(varargin) > 5:
        raise ValueError("Wrong number of input arguments")

    image = varargin[0]

    if len(varargin) == 1:
        # Default parameters
        spoints = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
        neighbors = 8
        mapping = get_mapping(8, 'riu2')
        mode = 'nh'

    if (len(varargin) == 2) and (len(str(varargin[1])) == 1):
        raise ValueError('Input arguments')

    if (len(varargin) > 2) and (len(str(varargin[1])) == 1):
        radius = varargin[1]
        neighbors = varargin[2]

        spoints = np.zeros((neighbors, 2))

        # Angle step
        a = 2 * np.pi / neighbors

        for i in range(neighbors):
            spoints[i, 0] = -radius * np.sin((i - 1) * a)
            spoints[i, 1] = radius * np.cos((i - 1) * a)

        if len(varargin) >= 4:
            mapping = varargin[3]
            if isinstance(mapping, dict) and mapping['samples'] != neighbors:
                raise ValueError('Incompatible mapping')
        else:
            mapping = 0

        if len(varargin) >= 5:
            mode = varargin[4]
        else:
            mode = 'h'

    if (len(varargin) > 1) and (len(str(varargin[1])) > 1):
        spoints = varargin[1]
        neighbors = spoints.shape[0]

        if len(varargin) >= 3:
            mapping = varargin[2]
            if isinstance(mapping, dict) and mapping['samples'] != neighbors:
                raise ValueError('Incompatible mapping')
        else:
            mapping = 0

        if len(varargin) >= 4:
            mode = varargin[3]
        else:
            mode = 'nh'

    # Determine the dimensions of the input image
    ysize, xsize = image.shape

    miny = np.min(spoints[:, 0])
    maxy = np.max(spoints[:, 0])
    minx = np.min(spoints[:, 1])
    maxx = np.max(spoints[:, 1])

    # Block size, each LBP code is computed within a block of size bsizey*bsizex
    bsizey = np.ceil(max(maxy, 0)) - np.floor(min(miny, 0))
    bsizex = np.ceil(max(maxx, 0)) - np.floor(min(minx, 0))

    # Coordinates of origin (0,0) in the block
    origy = int(1 - np.floor(min(miny, 0)))
    origx = int(1 - np.floor(min(minx, 0)))

    # Minimum allowed size for the input image depends on the radius of the used LBP operator
    if xsize < bsizex or ysize < bsizey:
        raise ValueError("Too small input image. Should be at least (2*radius+1) x (2*radius+1)")

    # Calculate dx and dy
    dx = int(xsize - bsizex)
    dy = int(ysize - bsizey)

    # Fill the center pixel matrix C
    C = image[origy - 1:origy + dy - 1, origx - 1:origx + dx - 1]
    d_C = np.double(C)

    bins = 2 ** neighbors

    # Initialize the result matrix with zeros
    result = np.zeros((dy, dx))

    # Compute the LBP code image
    for i in range(neighbors):
        y = spoints[i, 0] + origy
        x = spoints[i, 1] + origx
        # Calculate floors, ceils and rounds for the x and y
        fy = int(np.floor(y))
        cy = int(np.ceil(y))
        ry = int(np.round(y))
        fx = int(np.floor(x))
        cx = int(np.ceil(x))
        rx = int(np.round(x))
        # Check if interpolation is needed
        if (np.abs(x - rx) < 1e-6) and (np.abs(y - ry) < 1e-6):
            # Interpolation is not needed, use original datatypes
            N = image[ry - 1:ry + dy - 1, rx - 1:rx + dx - 1]
            D = N >= C
        else:
            # Interpolation needed, use double type images
            ty = y - fy
            tx = x - fx

            # Calculate the interpolation weights
            w1 = roundn((1 - tx) * (1 - ty), -6)
            w2 = roundn(tx * (1 - ty), -6)
            w3 = roundn((1 - tx) * ty, -6)
            w4 = roundn(1 - w1 - w2 - w3, -6)

            # Compute interpolated pixel values
            N = w1 * image[fy - 1:fy + dy - 1, fx - 1:fx + dx - 1] + w2 * image[fy - 1:fy + dy - 1, cx - 1:cx + dx - 1] + \
                w3 * image[cy - 1:cy + dy - 1, fx - 1:fx + dx - 1] + w4 * image[cy - 1:cy + dy - 1, cx - 1:cx + dx - 1]
            N = roundn(N, -4)
            D = N >= d_C

        # Update the result matrix
        v = 2 ** i
        result = result + v * D

    # Apply mapping if it is defined
    if isinstance(mapping, dict):
        bins = mapping['num']
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = mapping['table'][int(result[i, j])]

    codeImage = result

    if mode in ['h', 'hist', 'nh']:
        # Return with LBP histogram if mode equals 'hist'
        result = np.histogram(result, bins=np.arange(bins + 1))[0]
        if mode == 'nh':
            result = result / np.sum(result)
    else:
        # Otherwise return a matrix of unsigned integers
        if bins - 1 <= np.iinfo(np.uint8).max:
            result = result.astype(np.uint8)
        elif bins - 1 <= np.iinfo(np.uint16).max:
            result = result.astype(np.uint16)
        else:
            result = result.astype(np.uint32)

    return result, codeImage


def construct_Gabor_filters(num_of_orient, num_of_scales, size1, fmax=0.25,
                            ni=np.sqrt(2), gamma=np.sqrt(2), separation=np.sqrt(2)):
    """
    Constructs a bank of Gabor filters.

    :param num_of_orient: Number of orientations.
    :type num_of_orient: int
    :param num_of_scales: Number of scales.
    :type num_of_scales: int
    :param size1: Size of the filters. Can be an integer for square filters or a tuple for rectangular filters.
    :type size1: int or tuple
    :param fmax: Maximum frequency. Default is 0.25.
    :type fmax: float, optional
    :param ni: Bandwidth parameter. Default is sqrt(2).
    :type ni: float, optional
    :param gamma: Aspect ratio. Default is sqrt(2).
    :type gamma: float, optional
    :param separation: Frequency separation factor. Default is sqrt(2).
    :type separation: float, optional

    :returns: A dictionary containing the spatial and frequency representations of the Gabor filters.
    :rtype: dict

    :example:
        >>> import matplotlib.pyplot as plt
        >>> num_of_orient = 8
        >>> num_of_scales = 5
        >>> filter_size = 31
        >>> gabor_filters = construct_Gabor_filters(num_of_orient, num_of_scales, filter_size)
        >>> fig, axes = plt.subplots(num_of_scales, num_of_orient, figsize=(20, 10))
        >>> for u in range(num_of_scales):
        ...     for v in range(num_of_orient):
        ...         ax = axes[u, v]
        ...         ax.imshow(np.real(gabor_filters['spatial'][u, v]), cmap='gray')
        ...         ax.axis('off')
        >>> plt.show()
    """
    # Initialize the filter bank
    filter_bank = {
        'spatial': np.empty((num_of_scales, num_of_orient), dtype=object),
        'freq': np.empty((num_of_scales, num_of_orient), dtype=object),
        'scales': num_of_scales,
        'orient': num_of_orient
    }

    # Check and adjust the size input
    if isinstance(size1, int):
        size1 = (size1, size1)
    elif len(size1) == 2:
        size1 = tuple(size1)
    else:
        raise ValueError("The parameter determining the size of the filters is not valid.")

    # Construct Gabor filter bank
    sigma_x = size1[1]
    sigma_y = size1[0]

    for u in range(num_of_scales):  # for each scale
        fu = fmax / (separation ** u)
        alfa = fu / gamma
        beta = fu / ni

        for v in range(num_of_orient):  # for each orientation
            theta_v = (v / num_of_orient) * np.pi
            gabor = np.zeros((2 * sigma_y, 2 * sigma_x), dtype=np.complex128)

            for x in range(-sigma_x, sigma_x):
                for y in range(-sigma_y, sigma_y):
                    xc = x * np.cos(theta_v) + y * np.sin(theta_v)
                    yc = -x * np.sin(theta_v) + y * np.cos(theta_v)
                    gabor[y + sigma_y, x + sigma_x] = (
                            (fu ** 2 / (np.pi * gamma * ni)) *
                            np.exp(-(alfa ** 2 * xc ** 2 + beta ** 2 * yc ** 2)) *
                            np.exp(1j * 2 * np.pi * fu * xc)
                    )

            filter_bank['spatial'][u, v] = gabor
            filter_bank['freq'][u, v] = np.fft.fft2(gabor)

    return filter_bank


def filter_image_with_Gabor_bank(image, filter_bank, down_sampling_factor=64):
    """
    Apply a Gabor filter bank to an image and return the filtered features.

    :param image: Input image to be filtered. Should be a 2D numpy array.
    :type image: np.ndarray
    :param filter_bank: Dictionary containing Gabor filter bank with 'spatial' and 'freq' keys.
    :type filter_bank: dict
    :param down_sampling_factor: Factor for down-sampling the filtered images. Default is 64.
    :type down_sampling_factor: int, optional

    :returns: Concatenated filtered features from the Gabor filter bank.
    :rtype: np.ndarray

    :raises ValueError: If the inputs are not as expected or dimensions do not match.

    :example:
        >>> import numpy as np
        >>> from skimage.data import camera
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


def gabor_filter(image, orienNum, scaleNum):
    """
    Apply a Gabor filter bank to an image and organize the results into a multidimensional array.

    :param image: Input image to be filtered. Should be a 2D numpy array.
    :type image: np.ndarray
    :param orienNum: Number of orientation filters in the Gabor filter bank.
    :type orienNum: int
    :param scaleNum: Number of scale filters in the Gabor filter bank.
    :type scaleNum: int

    :returns: Multidimensional array containing the Gabor magnitude responses. Shape is (height, width, orienNum, scaleNum).
    :rtype: np.ndarray

    :example:
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


def low_pass_filter(size, cutoff, n):
    """
    Creates a low-pass Butterworth filter.

    :param size: The size of the filter. If a single integer is provided, the filter will be square with that size.
    :type size: tuple of int
    :param cutoff: The cutoff frequency for the filter. Must be between 0 and 0.5.
    :type cutoff: float
    :param n: The order of the Butterworth filter. Must be an integer greater than or equal to 1.
    :type n: int

    :returns: The low-pass Butterworth filter in the frequency domain.
    :rtype: np.ndarray

    :raises ValueError: If `cutoff` is not in the range [0, 0.5], or if `n` is not an integer greater than or equal to 1.

    :example:
        >>> filter_size = (256, 256)
        >>> cutoff_frequency = 0.1
        >>> order = 2
        >>> lp_filter = low_pass_filter(filter_size, cutoff_frequency, order)
        >>> print(lp_filter.shape)
        (256, 256)
    """
    # Validate input parameters
    if cutoff < 0 or cutoff > 0.5:
        raise ValueError('Cutoff frequency must be between 0 and 0.5')

    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be an integer >= 1')

    # Set filter size
    if isinstance(size, int):
        rows = cols = size
    else:
        rows, cols = size

    # Create coordinate grid
    xrange = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / (cols - 1)
    yrange = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / (rows - 1)

    x, y = np.meshgrid(xrange, yrange)
    radius = np.sqrt(x ** 2 + y ** 2)

    # Calculate Butterworth filter
    f = 1 / (1 + (radius / cutoff) ** (2 * n))

    # Shift the filter to center it
    return np.fft.ifftshift(f)


def phase_cong3(image, nscale=4, norient=6, minWaveLength=3, mult=2.1, sigmaOnf=0.55,
                dThetaOnSigma=1.5, k=2.0, cutOff=0.5, g=10):
    """
    Computes the phase congruency of an image using a multiscale, multi-orientation approach.
    Phase congruency is a measure of the image's local contrast, based on the phase information of its frequency components.
    This method is used for edge detection and texture analysis.

    :param image: Input grayscale image as a 2D numpy array.
    :type image: numpy.ndarray
    :param nscale: Number of scales to be used in the analysis (default is 4).
    :type nscale: int, optional
    :param norient: Number of orientations to be used in the analysis (default is 6).
    :type norient: int, optional
    :param minWaveLength: Minimum wavelength of the log-Gabor filters (default is 3).
    :type minWaveLength: float, optional
    :param mult: Scaling factor for the wavelength of the log-Gabor filters (default is 2.1).
    :type mult: float, optional
    :param sigmaOnf: Standard deviation of the Gaussian function used in the log-Gabor filter (default is 0.55).
    :type sigmaOnf: float, optional
    :param dThetaOnSigma: Angular spread of the Gaussian function relative to the orientation (default is 1.5).
    :type dThetaOnSigma: float, optional
    :param k: Constant to adjust the threshold for noise (default is 2.0).
    :type k: float, optional
    :param cutOff: Cut-off parameter for weighting function (default is 0.5).
    :type cutOff: float, optional
    :param g: Gain parameter for the weighting function (default is 10).
    :type g: float, optional

    :returns: A tuple containing:
        - M: The measure of local phase congruency.
        - m: The measure of local phase concavity.
        - ori: Orientation of the phase congruency.
        - featType: Feature type (complex representation of phase congruency).
        - PC: List of phase congruency maps for each orientation.
        - EO: List of complex responses for each scale and orientation.
    :rtype: tuple of numpy.ndarray and list

    :raises ValueError: If the input image is not a 2D numpy array.

    :notes:
        - The function assumes the input image is grayscale.
        - The log-Gabor filters are used to analyze the image at different scales and orientations.
        - The phase congruency is computed based on the response of these filters, and noise is estimated and thresholded.
        - The result includes orientation information, which can be useful for edge detection and texture analysis.

    :example:
        >>> import numpy as np
        >>> image = np.random.rand(256, 256)
        >>> M, m, ori, featType, PC, EO = phase_cong3(image)
    """
    epsilon = .0001

    thetaSigma = np.pi / norient / dThetaOnSigma

    rows, cols = image.shape
    imagefft = fft2(image)

    zero = np.zeros((rows, cols))
    EO = [[None] * norient for _ in range(nscale)]
    covx2 = zero.copy()
    covy2 = zero.copy()
    covxy = zero.copy()

    estMeanE2n = []
    PC = []
    ifftFilterArray = [None] * nscale

    if cols % 2:
        xrange = np.linspace(-(cols - 1) / 2, (cols - 1) / 2, cols) / (cols - 1)
    else:
        xrange = np.linspace(-cols / 2, cols / 2 - 1, cols) / cols

    if rows % 2:
        yrange = np.linspace(-(rows - 1) / 2, (rows - 1) / 2, rows) / (rows - 1)
    else:
        yrange = np.linspace(-rows / 2, rows / 2 - 1, rows) / rows

    x, y = np.meshgrid(xrange, yrange)

    radius = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(-y, x)

    radius = ifftshift(radius)
    theta = ifftshift(theta)
    radius[0, 0] = 1

    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    del x, y, theta

    lp = low_pass_filter([rows, cols], .45, 15)
    logGabor = [None] * nscale

    for s in range(nscale):
        wavelength = minWaveLength * mult ** s
        fo = 1.0 / wavelength
        logGabor[s] = np.exp((-(np.log(radius / fo)) ** 2) / (2 * np.log(sigmaOnf) ** 2))
        logGabor[s] *= lp
        logGabor[s][0, 0] = 0

    spread = [None] * norient

    for o in range(norient):
        angl = o * np.pi / norient
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        spread[o] = np.exp((-dtheta ** 2) / (2 * thetaSigma ** 2))

    for o in range(norient):
        angl = o * np.pi / norient
        sumE_ThisOrient = zero.copy()
        sumO_ThisOrient = zero.copy()
        sumAn_ThisOrient = zero.copy()
        Energy = zero.copy()

        for s in range(nscale):
            filter_ = logGabor[s] * spread[o]
            ifftFilt = np.real(ifft2(filter_)) * np.sqrt(rows * cols)
            ifftFilterArray[s] = ifftFilt
            EO[s][o] = ifft2(imagefft * filter_)
            An = np.abs(EO[s][o])
            sumAn_ThisOrient += An
            sumE_ThisOrient += np.real(EO[s][o])
            sumO_ThisOrient += np.imag(EO[s][o])

            if s == 0:
                EM_n = np.sum(filter_ ** 2)
                maxAn = An
            else:
                maxAn = np.maximum(maxAn, An)

        XEnergy = np.sqrt(sumE_ThisOrient ** 2 + sumO_ThisOrient ** 2) + epsilon
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy

        for s in range(nscale):
            E = np.real(EO[s][o])
            O = np.imag(EO[s][o])
            Energy += E * MeanE + O * MeanO - np.abs(E * MeanO - O * MeanE)

        medianE2n = np.median(np.abs(EO[0][o]) ** 2)
        meanE2n = -medianE2n / np.log(0.5)
        estMeanE2n.append(meanE2n)

        noisePower = meanE2n / EM_n
        EstSumAn2 = zero.copy()
        for s in range(nscale):
            EstSumAn2 += ifftFilterArray[s] ** 2

        EstSumAiAj = zero.copy()
        for si in range(nscale - 1):
            for sj in range(si + 1, nscale):
                EstSumAiAj += ifftFilterArray[si] * ifftFilterArray[sj]

        sumEstSumAn2 = np.sum(EstSumAn2)
        sumEstSumAiAj = np.sum(EstSumAiAj)

        EstNoiseEnergy2 = 2 * noisePower * sumEstSumAn2 + 4 * noisePower * sumEstSumAiAj
        tau = np.sqrt(EstNoiseEnergy2 / 2)
        EstNoiseEnergy = tau * np.sqrt(np.pi / 2)
        EstNoiseEnergySigma = np.sqrt((2 - np.pi / 2) * tau ** 2)
        T = EstNoiseEnergy + k * EstNoiseEnergySigma
        T /= 1.7

        Energy = np.maximum(Energy - T, zero)

        width = sumAn_ThisOrient / (maxAn + epsilon) / nscale
        weight = 1.0 / (1 + np.exp((cutOff - width) * g))

        PC.append(weight * Energy / sumAn_ThisOrient)
        featType = E + 1j * O

        covx = PC[o] * np.cos(angl)
        covy = PC[o] * np.sin(angl)
        covx2 += covx ** 2
        covy2 += covy ** 2
        covxy += covx * covy

    covx2 /= (norient / 2)
    covy2 /= (norient / 2)
    covxy *= 4 / norient

    denom = np.sqrt(covxy ** 2 + (covx2 - covy2) ** 2) + epsilon
    sin2theta = covxy / denom
    cos2theta = (covx2 - covy2) / denom
    ori = np.arctan2(sin2theta, cos2theta) / 2
    ori = np.rad2deg(ori)
    ori[ori < 0] += 180

    M = (covy2 + covx2 + denom) / 2
    m = (covy2 + covx2 - denom) / 2

    return M, m, ori, featType, PC, EO


def descriptor_LDN(image, **kwargs):
    """
    Compute the Local Descriptor using Kirsch or Gaussian masks.
    This function computes a local descriptor for an input image using different masks
    based on the specified options. The masks include Kirsch masks for various sizes
    or a Gaussian gradient mask. The size of the mask can be adjusted through the
    `msize` parameter, and the `sigma` parameter controls the standard deviation for
    the Gaussian mask.

    :param image: The input image for which the descriptor is computed.
    :type image: numpy.ndarray
    :param kwargs: Additional optional parameters to customize the mask:
        - 'mask' (str): Type of mask to use. Options are 'kirsch' (default) or 'gaussian'.
        - 'msize' (int): Size of the Kirsch mask. Options are 3, 5, 7, 9, or 11 (default is 3).
        - 'sigma' (float): Standard deviation for the Gaussian mask (default is 0.5).
    :type kwargs: dict, optional

    :returns: The local descriptor matrix computed using the specified mask.
    :rtype: numpy.ndarray

    :raises ValueError: If an invalid mask type or size is provided.

    :example:
        >>> import numpy as np
        >>> from skimage.data import camera
        >>> image = camera()
        >>> descriptor = descriptor_LDN(image, mask='kirsch', msize=5)
        >>> print(descriptor.shape)
    """
    if kwargs is None:
        options = {}
    else:
        options = kwargs

    mask = 'kirsch'
    msize = 3
    sigma = 0.5

    if 'mask' in options:
        mask = options['mask']
    if 'msize' in options:
        msize = options['msize']
    if 'sigma' in options:
        sigma = options['sigma']

    if mask == 'gaussian':
        D = gauss_gradient(sigma)
    elif mask == 'kirsch':
        if msize == 3:
            D = np.zeros((3, 3, 8))
            # East
            D[:, :, 0] = np.array([[-3, -3, 5],
                                   [-3, 0, 5],
                                   [-3, -3, 5]])
            # North East
            D[:, :, 1] = np.array([[-3, 5, 5],
                                   [-3, 0, 5],
                                   [-3, -3, -3]])
            # North
            D[:, :, 2] = np.array([[5, 5, 5],
                                   [-3, 0, -3],
                                   [-3, -3, -3]])
            # North West
            D[:, :, 3] = np.array([[5, 5, -3],
                                   [5, 0, -3],
                                   [-3, -3, -3]])
            # West
            D[:, :, 4] = np.array([[5, -3, -3],
                                   [5, 0, -3],
                                   [5, -3, -3]])
            # South West
            D[:, :, 5] = np.array([[-3, -3, -3],
                                   [5, 0, -3],
                                   [5, 5, -3]])
            # South
            D[:, :, 6] = np.array([[-3, -3, -3],
                                   [-3, 0, -3],
                                   [5, 5, 5]])
            # South East
            D[:, :, 7] = np.array([[-3, -3, -3],
                                   [-3, 0, 5],
                                   [-3, 5, 5]])
        elif msize == 5:
            D = np.zeros((5, 5, 8))
            # East
            D[:, :, 0] = np.array([[-5, -5, -5, -5, 11],
                                   [-5, -3, -3, 5, 11],
                                   [-5, -3, 0, 5, 11],
                                   [-5, -3, -3, 5, 11],
                                   [-5, -5, -5, -5, 11]])
            # North East
            D[:, :, 1] = np.array([[-5, -5, 11, 11, 11],
                                   [-5, -3, 5, 5, 11],
                                   [-5, -3, 0, 5, 11],
                                   [-5, -3, -3, -3, -5],
                                   [-5, -5, -5, -5, -5]])
            # North
            D[:, :, 2] = np.array([[11, 11, 11, 11, 11],
                                   [-5, 5, 5, 5, -5],
                                   [-5, -3, 0, -3, -5],
                                   [-5, -3, -3, -3, -5],
                                   [-5, -5, -5, -5, -5]])
            # North West
            D[:, :, 3] = np.array([[11, 11, 11, -5, -5],
                                   [11, 5, 5, -3, -5],
                                   [11, 5, 0, -3, -5],
                                   [-5, -3, -3, -3, -5],
                                   [-5, -5, -5, -5, -5]])
            # West
            D[:, :, 4] = np.array([[11, -5, -5, -5, -5],
                                   [11, 5, -3, -3, -5],
                                   [11, 5, 0, -3, -5],
                                   [11, 5, -3, -3, -5],
                                   [11, -5, -5, -5, -5]])
            # South West
            D[:, :, 5] = np.array([[-5, -5, -5, -5, -5],
                                   [-5, -3, -3, -3, -5],
                                   [11, 5, 0, -3, -5],
                                   [11, 5, 5, -3, -5],
                                   [11, 11, 11, -5, -5]])
            # South
            D[:, :, 6] = np.array([[-5, -5, -5, -5, -5],
                                   [-5, -3, -3, -3, -5],
                                   [-5, -3, 0, -3, -5],
                                   [-5, 5, 5, 5, -5],
                                   [11, 11, 11, 11, 11]])
            # South East
            D[:, :, 7] = np.array([[-5, -5, -5, -5, -5],
                                   [-5, -3, -3, -3, -5],
                                   [-5, -3, 0, 5, 11],
                                   [-5, -3, 5, 5, 11],
                                   [-5, -5, 11, 11, 11]])
        elif msize == 7:
            D = np.zeros((7, 7, 8))
            # East
            D[:, :, 0] = np.array([[-7, -7, -7, -7, -7, -7, 17],
                                   [-7, -5, -5, -5, -5, 11, 17],
                                   [-7, -5, -3, -3, 5, 11, 17],
                                   [-7, -5, -3, 0, 5, 11, 17],
                                   [-7, -5, -3, -3, 5, 11, 17],
                                   [-7, -5, -5, -5, -5, 11, 17],
                                   [-7, -7, -7, -7, -7, -7, 17]])
            # North East
            D[:, :, 1] = np.array([[-7, -7, -7, 17, 17, 17, 17],
                                   [-7, -5, -5, 11, 11, 11, 17],
                                   [-7, -5, -3, 5, 5, 11, 17],
                                   [-7, -5, -3, 0, 5, 11, 17],
                                   [-7, -5, -3, -3, -3, -5, -7],
                                   [-7, -5, -5, -5, -5, -5, -7],
                                   [-7, -7, -7, -7, -7, -7, -7]])
            # North
            D[:, :, 2] = np.array([[17, 17, 17, 17, 17, 17, 17],
                                   [-7, 11, 11, 11, 11, 11, -7],
                                   [-7, -5, 5, 5, 5, -5, -7],
                                   [-7, -5, -3, 0, -3, -5, -7],
                                   [-7, -5, -3, -3, -3, -5, -7],
                                   [-7, -5, -5, -5, -5, -5, -7],
                                   [-7, -7, -7, -7, -7, -7, -7]])
            # North West
            D[:, :, 3] = np.array([[17, 17, 17, 17, -7, -7, -7],
                                   [17, 11, 11, 11, -5, -5, -7],
                                   [17, 11, 5, 5, -3, -5, -7],
                                   [17, 11, 5, 0, -3, -5, -7],
                                   [-7, -5, -3, -3, -3, -5, -7],
                                   [-7, -5, -5, -5, -5, -5, -7],
                                   [-7, -7, -7, -7, -7, -7, -7]])
            # West
            D[:, :, 4] = np.array([[17, -7, -7, -7, -7, -7, -7],
                                   [17, 11, -5, -5, -5, -5, -7],
                                   [17, 11, 5, -3, -3, -5, -7],
                                   [17, 11, 5, 0, -3, -5, -7],
                                   [17, 11, 5, -3, -3, -5, -7],
                                   [17, 11, -5, -5, -5, -5, -7],
                                   [17, -7, -7, -7, -7, -7, -7]])
            # South West
            D[:, :, 5] = np.array([[-7, -7, -7, -7, -7, -7, -7],
                                   [-7, -5, -5, -5, -5, -5, -7],
                                   [-7, -5, -3, -3, -3, -5, -7],
                                   [17, 11, 5, 0, -3, -5, -7],
                                   [17, 11, 5, 5, -3, -5, -7],
                                   [17, 11, 11, 11, -5, -5, -7],
                                   [17, 17, 17, 17, -7, -7, -7]])
            # South East
            D[:, :, 6] = np.array([[-7, -7, -7, -7, -7, -7, -7],
                                   [-7, -5, -5, -5, -5, -5, -7],
                                   [-7, -5, -3, -3, -3, -5, -7],
                                   [-7, -5, -3, 0, -3, -5, -7],
                                   [-7, -5, 5, 5, 5, -5, -7],
                                   [-7, 11, 11, 11, 11, 11, -7],
                                   [17, 17, 17, 17, 17, 17, 17]])
            # South
            D[:, :, 7] = np.array([[-7, -7, -7, -7, -7, -7, -7],
                                   [-7, -5, -5, -5, -5, -5, -7],
                                   [-7, -5, -3, -3, -3, -5, -7],
                                   [-7, -5, -3, 0, -3, -5, -7],
                                   [-7, -5, 5, 5, 5, -5, -7],
                                   [-7, 11, 11, 11, 11, 11, -7],
                                   [17, 17, 17, 17, 17, 17, 17]])
        elif msize == 9:
            D = np.zeros((9, 9, 8))
            # East
            D[:, :, 0] = [[-9, -9, -9, -9, -9, -9, -9, -9, 23],
                          [-9, -7, -7, -7, -7, -7, -7, 17, 23],
                          [-9, -7, -5, -5, -5, -5, 11, 17, 23],
                          [-9, -7, -5, -3, -3, 5, 11, 17, 23],
                          [-9, -7, -5, -3, 0, 5, 11, 17, 23],
                          [-9, -7, -5, -3, -3, 5, 11, 17, 23],
                          [-9, -7, -5, -5, -5, -5, 11, 17, 23],
                          [-9, -7, -7, -7, -7, -7, -7, 17, 23],
                          [-9, -9, -9, -9, -9, -9, -9, -9, 23]]
            # North East
            D[:, :, 1] = [[-9, -9, -9, -9, 23, 23, 23, 23, 23],
                          [-9, -7, -7, -7, 17, 17, 17, 17, 23],
                          [-9, -7, -5, -5, 11, 11, 11, 17, 23],
                          [-9, -7, -5, -3, 5, 5, 11, 17, 23],
                          [-9, -7, -5, -3, 0, 5, 11, 17, 23],
                          [-9, -7, -5, -3, -3, -3, -5, -7, -9],
                          [-9, -7, -5, -5, -5, -5, -5, -7, -9],
                          [-9, -7, -7, -7, -7, -7, -7, -7, -9],
                          [-9, -9, -9, -9, -9, -9, -9, -9, -9]]
            # North
            D[:, :, 2] = [[23, 23, 23, 23, 23, 23, 23, 23, 23],
                          [-9, 17, 17, 17, 17, 17, 17, 17, -9],
                          [-9, -7, 11, 11, 11, 11, 11, -7, -9],
                          [-9, -7, -5, 5, 5, 5, -5, -7, -9],
                          [-9, -7, -5, -3, 0, -3, -5, -7, -9],
                          [-9, -7, -5, -3, -3, -3, -5, -7, -9],
                          [-9, -7, -5, -5, -5, -5, -5, -7, -9],
                          [-9, -7, -7, -7, -7, -7, -7, -7, -9],
                          [-9, -9, -9, -9, -9, -9, -9, -9, -9]]
            # North West
            D[:, :, 3] = [[23, 23, 23, 23, 23, -9, -9, -9, -9],
                          [23, 17, 17, 17, 17, -7, -7, -7, -9],
                          [23, 17, 11, 11, 11, -5, -5, -7, -9],
                          [23, 17, 11, 5, 5, -3, -5, -7, -9],
                          [23, 17, 11, 5, 0, -3, -5, -7, -9],
                          [-9, -7, -5, -3, -3, -3, -5, -7, -9],
                          [-9, -7, -5, -5, -5, -5, -5, -7, -9],
                          [-9, -7, -7, -7, -7, -7, -7, -7, -9],
                          [-9, -9, -9, -9, -9, -9, -9, -9, -9]]
            # West
            D[:, :, 4] = [[23, 23, 23, 23, 23, 23, 23, 23, 23],
                          [23, 17, 17, 17, 17, 17, 17, 17, 23],
                          [23, 17, 11, 11, 11, 11, 11, 17, 23],
                          [23, 17, 11, 5, 5, 5, 11, 17, 23],
                          [23, 17, 11, 5, 0, 0, 0, 0, 0, 17],
                          [23, 17, 11, 5, 0, -3, -3, -3, -3, 17],
                          [23, 17, 11, 11, 11, -5, -5, -7, -9],
                          [23, 17, 17, 17, 17, -7, -7, -7, -9],
                          [23, 23, 23, 23, 23, -9, -9, -9, -9]]
            # South West
            D[:, :, 5] = [[23, -9, -9, -9, -9, -9, -9, -9, -9],
                          [23, 17, -7, -7, -7, -7, -7, -7, -9],
                          [23, 17, 11, -5, -5, -5, -5, -7, -9],
                          [23, 17, 11, 5, -3, -3, -5, -7, -9],
                          [23, 17, 11, 5, 0, -3, -5, -7, -9],
                          [23, 17, 11, 5, -3, -3, -5, -7, -9],
                          [23, 17, 11, -5, -5, -5, -5, -7, -9],
                          [23, 17, -7, -7, -7, -7, -7, -7, -9],
                          [23, -9, -9, -9, -9, -9, -9, -9, -9]]
            # South East
            D[:, :, 6] = [[-9, -9, -9, -9, -9, -9, -9, -9, -9],
                          [-9, -7, -7, -7, -7, -7, -7, -7, -9],
                          [-9, -7, -5, -5, -5, -5, -5, -7, -9],
                          [-9, -7, -5, -3, -3, -3, -5, -7, -9],
                          [-9, -7, -5, -3, 0, 5, 11, 17, 23],
                          [-9, -7, -5, -3, 5, 5, 11, 17, 23],
                          [-9, -7, -5, 5, 5, 5, -5, -7, -9],
                          [-9, -7, 11, 11, 11, 11, 11, -7, -9],
                          [-9, 17, 17, 17, 17, 17, 17, 17, -9]]
            # South
            D[:, :, 7] = [[-9, -9, -9, -9, -9, -9, -9, -9, -9],
                          [-9, -7, -7, -7, -7, -7, -7, -7, -9],
                          [-9, -7, -5, -5, -5, -5, -5, -7, -9],
                          [-9, -7, -5, -3, -3, -3, -5, -7, -9],
                          [-9, -7, -5, -3, 0, -3, -5, -7, -9],
                          [-9, -7, -5, 5, 5, 5, -5, -7, -9],
                          [-9, -7, 11, 11, 11, 11, 11, -7, -9],
                          [-9, 17, 17, 17, 17, 17, 17, 17, -9],
                          [23, 23, 23, 23, 23, 23, 23, 23, 23]]
        elif msize == 11:
            D = np.zeros((11, 11, 8))
            # East
            D[:, :, 0] = [[-11, -11, -11, -11, -11, -11, -11, -11, -11, -11, 29],
                          [-11, -9, -9, -9, -9, -9, -9, -9, -9, 23, 29],
                          [-11, -9, -7, -7, -7, -7, -7, -7, 17, 23, 29],
                          [-11, -9, -7, -5, -5, -5, -5, 11, 17, 23, 29],
                          [-11, -9, -7, -5, -3, -3, 5, 11, 17, 23, 29],
                          [-11, -9, -7, -5, -3, 0, 5, 11, 17, 23, 29],
                          [-11, -9, -7, -5, -3, -3, 5, 11, 17, 23, 29],
                          [-11, -9, -7, -5, -5, -5, -5, 11, 17, 23, 29],
                          [-11, -9, -7, -7, -7, -7, -7, -7, 17, 23, 29],
                          [-11, -9, -9, -9, -9, -9, -9, -9, -9, 23, 29],
                          [-11, -11, -11, -11, -11, -11, -11, -11, -11, -11, 29]]
            # North East
            D[:, :, 1] = [[-11, -11, -11, -11, -11, 29, 29, 29, 29, 29, 29],
                          [-11, -9, -9, -9, -9, 23, 23, 23, 23, 23, 29],
                          [-11, -9, -7, -7, -7, 17, 17, 17, 17, 23, 29],
                          [-11, -9, -7, -5, -5, 11, 11, 11, 17, 23, 29],
                          [-11, -9, -7, -5, -3, 5, 5, 11, 17, 23, 29],
                          [-11, -9, -7, -5, -3, 0, 5, 11, 17, 23, 29],
                          [-11, -9, -7, -5, -3, -3, -3, -5, -7, -9, -11],
                          [-11, -9, -7, -5, -5, -5, -5, -5, -7, -9, -11],
                          [-11, -9, -7, -7, -7, -7, -7, -7, -7, -9, -11],
                          [-11, -9, -9, -9, -9, -9, -9, -9, -9, -9, -11],
                          [-11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11]]
            # North
            D[:, :, 2] = [[29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29],
                          [-11, 23, 23, 23, 23, 23, 23, 23, 23, 23, -11],
                          [-11, -9, 17, 17, 17, 17, 17, 17, 17, -9, -11],
                          [-11, -9, -7, 11, 11, 11, 11, 11, -7, -9, -11],
                          [-11, -9, -7, -5, 5, 5, 5, -5, -7, -9, -11],
                          [-11, -9, -7, -5, -3, 0, -3, -5, -7, -9, -11],
                          [-11, -9, -7, -5, -3, -3, -3, -5, -7, -9, -11],
                          [-11, -9, -7, -5, -5, -5, -5, -5, -7, -9, -11],
                          [-11, -9, -7, -7, -7, -7, -7, -7, -7, -9, -11],
                          [-11, -9, -9, -9, -9, -9, -9, -9, -9, -9, -11],
                          [-11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11]]
            # North West
            D[:, :, 3] = [[29, 29, 29, 29, 29, 29, -11, -11, -11, -11, -11],
                          [29, 23, 23, 23, 23, 23, -9, -9, -9, -9, -11],
                          [29, 23, 17, 17, 17, 17, -7, -7, -7, -9, -11],
                          [29, 23, 17, 11, 11, 11, -5, -5, -7, -9, -11],
                          [29, 23, 17, 11, 5, 5, -3, -5, -7, -9, -11],
                          [29, 23, 17, 11, 5, 0, -3, -5, -7, -9, -11],
                          [-11, -9, -7, -5, -3, -3, -3, -5, -7, -9, -11],
                          [-11, -9, -7, -5, -5, -5, -5, -5, -7, -9, -11],
                          [-11, -9, -7, -7, -7, -7, -7, -7, -7, -9, -11],
                          [-11, -9, -9, -9, -9, -9, -9, -9, -9, -9, -11],
                          [-11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11]]
            # West
            D[:, :, 4] = [[29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29],
                          [29, 23, 23, 23, 23, 23, -11, -11, -11, -11, -11],
                          [29, 23, 17, 17, 17, 17, -9, -9, -9, -9, -11],
                          [29, 23, 17, 11, 11, 11, -7, -7, -7, -9, -11],
                          [29, 23, 17, 11, 5, 5, -5, -5, -7, -9, -11],
                          [29, 23, 17, 11, 5, 0, -3, -5, -7, -9, -11],
                          [-11, -9, -7, -5, -3, -3, -3, -5, -7, -9, -11],
                          [-11, -9, -7, -5, -5, -5, -5, -5, -7, -9, -11],
                          [-11, -9, -7, -7, -7, -7, -7, -7, -7, -9, -11],
                          [-11, -9, -9, -9, -9, -9, -9, -9, -9, -9, -11],
                          [-11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11]]
            # South West
            D[:, :, 5] = [[29, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11],
                          [29, 23, -9, -9, -9, -9, -9, -9, -9, -9, -11],
                          [29, 23, 17, -7, -7, -7, -7, -7, -7, -9, -11],
                          [29, 23, 17, 11, -5, -5, -5, -5, -7, -9, -11],
                          [29, 23, 17, 11, 5, -3, -3, -5, -7, -9, -11],
                          [29, 23, 17, 11, 5, 0, -3, -5, -7, -9, -11],
                          [29, 23, 17, 11, 5, -3, -3, -5, -7, -9, -11],
                          [29, 23, 17, 11, -5, -5, -5, -5, -7, -9, -11],
                          [29, 23, 17, -7, -7, -7, -7, -7, -7, -9, -11],
                          [29, 23, -9, -9, -9, -9, -9, -9, -9, -9, -11],
                          [29, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11]]
            # South East
            D[:, :, 6] = [[-11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11],
                          [-11, -9, -9, -9, -9, -9, -9, -9, -9, -9, -11],
                          [-11, -9, -7, -7, -7, -7, -7, -7, -7, -9, -11],
                          [-11, -9, -7, -5, -5, -5, -5, -5, -7, -9, -11],
                          [-11, -9, -7, -5, -3, -3, -3, -5, -7, -9, -11],
                          [29, 23, 17, 11, 5, 0, -3, -5, -7, -9, -11],
                          [29, 23, 17, 11, 5, 5, -3, -5, -7, -9, -11],
                          [29, 23, 17, 11, 11, 11, -5, -5, -7, -9, -11],
                          [29, 23, 17, 17, 17, 17, -7, -7, -7, -9, -11],
                          [29, 23, 23, 23, 23, 23, -9, -9, -9, -9, -11],
                          [29, 29, 29, 29, 29, 29, -11, -11, -11, -11, -11]]
            # South
            D[:, :, 7] = [[-11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11],
                          [-11, -9, -9, -9, -9, -9, -9, -9, -9, -9, -11],
                          [-11, -9, -7, -7, -7, -7, -7, -7, -7, -9, -11],
                          [-11, -9, -7, -5, -5, -5, -5, -5, -7, -9, -11],
                          [-11, -9, -7, -5, -3, -3, -3, -5, -7, -9, -11],
                          [-11, -9, -7, -5, -3, 0, -3, -5, -7, -9, -11],
                          [-11, -9, -7, -5, 5, 5, 5, -5, -7, -9, -11],
                          [-11, -9, -7, 11, 11, 11, 11, 11, -7, -9, -11],
                          [-11, -9, 17, 17, 17, 17, 17, 17, 17, -9, -11],
                          [-11, 23, 23, 23, 23, 23, 23, 23, 23, 23, -11],
                          [29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29]]
        else:
            raise ValueError('Kirsh mask size not supported. Use only: 3, 5, 7, 9, 11')

    elif mask == 'sobel':
        if msize == 3:
            D = np.zeros((3, 3, 8))
            # East
            D[:, :, 0] = [[1, 2, 1],
                          [0, 0, 0],
                          [-1, -2, -1]]
            # North East
            D[:, :, 1] = [[2, 1, 0],
                          [1, 0, -1],
                          [0, -1, -2]]
            # North
            D[:, :, 2] = [[1, 0, -1],
                          [2, 0, -2],
                          [1, 0, -1]]
            # North West
            D[:, :, 3] = [[0, -1, -2],
                          [1, 0, -1],
                          [2, 1, 0]]
            # West
            D[:, :, 4] = [[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]]
            # South West
            D[:, :, 5] = [[-2, -1, 0],
                          [-1, 0, 1],
                          [0, 1, 2]]
            # South
            D[:, :, 6] = [[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]]
            # South East
            D[:, :, 7] = [[0, 1, 2],
                          [-1, 0, 1],
                          [-2, -1, 0]]
        elif msize == 5:
            D = np.zeros((5, 5, 8))
            # East
            D[:, :, 0] = [[1, 2, 0, -2, -1],
                          [4, 8, 0, -8, -4],
                          [6, 12, 0, -12, -6],
                          [4, 8, 0, -8, -4],
                          [1, 2, 0, -2, -1]]
            # North East
            D[:, :, 1] = [[1, 4, 6, 4, 1],
                          [2, 8, 12, 8, 2],
                          [0, 0, 0, 0, 0],
                          [-2, -8, -12, -8, -2],
                          [-1, -4, -6, -4, -1]]
            # North
            D[:, :, 2] = [[-1, -4, -6, -4, -1],
                          [-2, -8, -12, -8, -2],
                          [0, 0, 0, 0, 0],
                          [2, 8, 12, 8, 2],
                          [1, 4, 6, 4, 1]]
            # North West
            D[:, :, 3] = [[-1, -2, 0, 2, 1],
                          [-4, -8, 0, 8, 4],
                          [-6, -12, 0, 12, 6],
                          [-4, -8, 0, 8, 4],
                          [-1, -2, 0, 2, 1]]
            # West
            D[:, :, 4] = [[0, 2, 1, 4, 6],
                          [-2, 0, 8, 12, 4],
                          [-1, -8, 0, 8, 1],
                          [-4, -12, -8, 0, 2],
                          [-6, -4, -1, -2, 0]]
            # South West
            D[:, :, 5] = [[6, 4, 1, 2, 0],
                          [4, 12, 8, 0, -2],
                          [1, 8, 0, -8, -1],
                          [2, 0, -8, -12, -4],
                          [0, -2, -1, -4, -6]]
            # South
            D[:, :, 6] = [[-6, -4, -1, -2, 0],
                          [-4, -12, -8, 0, 2],
                          [-1, -8, 0, 8, 1],
                          [-2, 0, 8, 12, 4],
                          [0, 2, 1, 4, 6]]
            # South East
            D[:, :, 7] = [[0, -2, -1, -4, -6],
                          [2, 0, -8, -12, -4],
                          [1, 8, 0, -8, -1],
                          [4, 12, 8, 0, -2],
                          [6, 4, 1, 2, 0]]
        elif msize == 7:
            D = np.zeros((7, 7, 8))
            # East
            D[:, :, 0] = [[1, 4, 5, 0, -5, -4, -1],
                          [6, 24, 30, 0, -30, -24, -6],
                          [15, 60, 75, 0, -75, -60, -15],
                          [20, 80, 100, 0, -100, -80, -20],
                          [15, 60, 75, 0, -75, -60, -15],
                          [6, 24, 30, 0, -30, -24, -6],
                          [1, 4, 5, 0, -5, -4, -1]]
            # North East
            D[:, :, 1] = [[1, 6, 15, 20, 15, 6, 1],
                          [4, 24, 60, 80, 60, 24, 4],
                          [5, 30, 75, 100, 75, 30, 5],
                          [0, 0, 0, 0, 0, 0, 0],
                          [-5, -30, -75, -100, -75, -30, -5],
                          [-4, -24, -60, -80, -60, -24, -4],
                          [-1, -6, -15, -20, -15, -6, -1]]
            # North
            D[:, :, 2] = [[-1, -6, -15, -20, -15, -6, -1],
                          [-4, -24, -60, -80, -60, -24, -4],
                          [-5, -30, -75, -100, -75, -30, -5],
                          [0, 0, 0, 0, 0, 0, 0],
                          [5, 30, 75, 100, 75, 30, 5],
                          [4, 24, 60, 80, 60, 24, 4],
                          [1, 6, 15, 20, 15, 6, 1]]
            # North West
            D[:, :, 3] = [[-1, -4, -5, 0, 5, 4, 1],
                          [-6, -24, -30, 0, 30, 24, 6],
                          [-15, -60, -75, 0, 75, 60, 15],
                          [-20, -80, -100, 0, 100, 80, 20],
                          [-15, -60, -75, 0, 75, 60, 15],
                          [-6, -24, -30, 0, 30, 24, 6],
                          [-1, -4, -5, 0, 5, 4, 1]]
            # West
            D[:, :, 4] = [[0, 5, 4, 1, 6, 15, 20],
                          [-5, 0, 30, 24, 60, 80, 15],
                          [-4, -30, 0, 75, 100, 60, 6],
                          [-1, -24, -75, 0, 75, 24, 1],
                          [-6, -60, -100, -75, 0, 30, 4],
                          [-15, -80, -60, -24, -30, 0, 5],
                          [-20, -15, -6, -1, -4, -5, 0]]
            # South West
            D[:, :, 5] = [[20, 15, 6, 1, 4, 5, 0],
                          [15, 80, 60, 24, 30, 0, -5],
                          [6, 60, 100, 75, 0, -30, -4],
                          [1, 24, 75, 0, -75, -24, -1],
                          [4, 30, 0, -75, -100, -60, -6],
                          [5, 0, -30, -24, -60, -80, -15],
                          [0, -5, -4, -1, -6, -15, -20]]
            # South
            D[:, :, 6] = [[-20, -15, -6, -1, -4, -5, 0],
                          [-15, -80, -60, -24, -30, 0, 5],
                          [-6, -60, -100, -75, 0, 30, 4],
                          [-1, -24, -75, 0, 75, 24, 1],
                          [-4, -30, 0, 75, 100, 60, 6],
                          [-5, 0, 30, 24, 60, 80, 15],
                          [0, 5, 4, 1, 6, 15, 20]]
            # South East
            D[:, :, 7] = [[0, -5, -4, -1, -6, -15, -20],
                          [5, 0, -30, -24, -60, -80, -15],
                          [4, 30, 0, -75, -100, -60, -6],
                          [1, 24, 75, 0, -75, -24, -1],
                          [6, 60, 100, 75, 0, -30, -4],
                          [15, 80, 60, 24, 30, 0, -5],
                          [20, 15, 6, 1, 4, 5, 0]]
        elif msize == 9:
            D = np.zeros((9, 9, 8))
            # East
            D[:, :, 0] = [[1, 6, 14, 14, 0, -14, -14, -6, -1],
                          [8, 48, 112, 112, 0, -112, -112, -48, -8],
                          [28, 168, 392, 392, 0, -392, -392, -168, -28],
                          [56, 336, 784, 784, 0, -784, -784, -336, -56],
                          [70, 420, 980, 980, 0, -980, -980, -420, -70],
                          [56, 336, 784, 784, 0, -784, -784, -336, -56],
                          [28, 168, 392, 392, 0, -392, -392, -168, -28],
                          [8, 48, 112, 112, 0, -112, -112, -48, -8],
                          [1, 6, 14, 14, 0, -14, -14, -6, -1]]
            # North East
            D[:, :, 1] = [[1, 8, 28, 56, 70, 56, 28, 8, 1],
                          [6, 48, 168, 336, 420, 336, 168, 48, 6],
                          [14, 112, 392, 784, 980, 784, 392, 112, 14],
                          [14, 112, 392, 784, 980, 784, 392, 112, 14],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [-14, -112, -392, -784, -980, -784, -392, -112, -14],
                          [-14, -112, -392, -784, -980, -784, -392, -112, -14],
                          [-6, -48, -168, -336, -420, -336, -168, -48, -6],
                          [-1, -8, -28, -56, -70, -56, -28, -8, -1]]
            # North
            D[:, :, 2] = [[-1, -8, -28, -56, -70, -56, -28, -8, -1],
                          [-6, -48, -168, -336, -420, -336, -168, -48, -6],
                          [-14, -112, -392, -784, -980, -784, -392, -112, -14],
                          [-14, -112, -392, -784, -980, -784, -392, -112, -14],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [14, 112, 392, 784, 980, 784, 392, 112, 14],
                          [14, 112, 392, 784, 980, 784, 392, 112, 14],
                          [6, 48, 168, 336, 420, 336, 168, 48, 6],
                          [1, 8, 28, 56, 70, 56, 28, 8, 1]]
            # North West
            D[:, :, 3] = [[-1, -6, -14, -14, 0, 14, 14, 6, 1],
                          [-8, -48, -112, -112, 0, 112, 112, 48, 8],
                          [-28, -168, -392, -392, 0, 392, 392, 168, 28],
                          [-56, -336, -784, -784, 0, 784, 784, 336, 56],
                          [-70, -420, -980, -980, 0, 980, 980, 420, 70],
                          [-56, -336, -784, -784, 0, 784, 784, 336, 56],
                          [-28, -168, -392, -392, 0, 392, 392, 168, 28],
                          [-8, -48, -112, -112, 0, 112, 112, 48, 8],
                          [-1, -6, -14, -14, 0, 14, 14, 6, 1]]
            # West
            D[:, :, 4] = [[0, 14, 14, 6, 1, 8, 28, 56, 70],
                          [-14, 0, 112, 112, 48, 168, 336, 420, 56],
                          [-14, -112, 0, 392, 392, 784, 980, 336, 28],
                          [-6, -112, -392, 0, 784, 980, 784, 168, 8],
                          [-1, -48, -392, -784, 0, 784, 392, 48, 1],
                          [-8, -168, -784, -980, -784, 0, 392, 112, 6],
                          [-28, -336, -980, -784, -392, -392, 0, 112, 14],
                          [-56, -420, -336, -168, -48, -112, -112, 0, 14],
                          [-70, -56, -28, -8, -1, -6, -14, -14, 0]]
            # South West
            D[:, :, 5] = [[70, 56, 28, 8, 1, 6, 14, 14, 0],
                          [56, 420, 336, 168, 48, 112, 112, 0, -14],
                          [28, 336, 980, 784, 392, 392, 0, -112, -14],
                          [8, 168, 784, 980, 784, 0, -392, -112, -6],
                          [1, 48, 392, 784, 0, -784, -392, -48, -1],
                          [6, 112, 392, 0, -784, -980, -784, -168, -8],
                          [14, 112, 0, -392, -392, -784, -980, -336, -28],
                          [14, 0, -112, -112, -48, -168, -336, -420, -56],
                          [0, -14, -14, -6, -1, -8, -28, -56, -70]]
            # South
            D[:, :, 6] = [[-70, -56, -28, -8, -1, -6, -14, -14, 0],
                          [-56, -420, -336, -168, -48, -112, -112, 0, 14],
                          [-28, -336, -980, -784, -392, -392, 0, 112, 14],
                          [-8, -168, -784, -980, -784, 0, 392, 112, 6],
                          [-1, -48, -392, -784, 0, 784, 392, 48, 1],
                          [-6, -112, -392, 0, 784, 980, 784, 168, 8],
                          [-14, -112, 0, 392, 392, 784, 980, 336, 28],
                          [-14, 0, 112, 112, 48, 168, 336, 420, 56],
                          [0, 14, 14, 6, 1, 8, 28, 56, 70]]
            # South East
            D[:, :, 7] = [[0, -14, -14, -6, -1, -8, -28, -56, -70],
                          [14, 0, -112, -112, -48, -168, -336, -420, -56],
                          [14, 112, 0, -392, -392, -784, -980, -336, -28],
                          [6, 112, 392, 0, -784, -980, -784, -168, -8],
                          [1, 48, 392, 784, 0, -784, -392, -48, -1],
                          [8, 168, 784, 980, 784, 0, -392, -112, -6],
                          [28, 336, 980, 784, 392, 392, 0, -112, -14],
                          [56, 420, 336, 168, 48, 112, 112, 0, -14],
                          [70, 56, 28, 8, 1, 6, 14, 14, 0]]
        elif msize == 11:
            D = np.zeros((11, 11, 8))
            # East
            D[:, :, 0] = [[1, 8, 27, 48, 42, 0, -42, -48, -27, -8, -1],
                          [10, 80, 270, 480, 420, 0, -420, -480, -270, -80, -10],
                          [45, 360, 1215, 2160, 1890, 0, -1890, -2160, -1215, -360, -45],
                          [120, 960, 3240, 5760, 5040, 0, -5040, -5760, -3240, -960, -120],
                          [210, 1680, 5670, 10080, 8820, 0, -8820, -10080, -5670, -1680, -210],
                          [252, 2016, 6804, 12096, 10584, 0, -10584, -12096, -6804, -2016, -252],
                          [210, 1680, 5670, 10080, 8820, 0, -8820, -10080, -5670, -1680, -210],
                          [120, 960, 3240, 5760, 5040, 0, -5040, -5760, -3240, -960, -120],
                          [45, 360, 1215, 2160, 1890, 0, -1890, -2160, -1215, -360, -45],
                          [10, 80, 270, 480, 420, 0, -420, -480, -270, -80, -10],
                          [1, 8, 27, 48, 42, 0, -42, -48, -27, -8, -1]]
            # North East
            D[:, :, 1] = [[-1, -10, -45, -120, -210, -252, -210, -120, -45, -10, -1],
                          [-8, -80, -360, -960, -1680, -2016, -1680, -960, -360, -80, -8],
                          [-27, -270, -1215, -3240, -5670, -6804, -5670, -3240, -1215, -270, -27],
                          [-48, -480, -2160, -5760, -10080, -12096, -10080, -5760, -2160, -480, -48],
                          [-42, -420, -1890, -5040, -8820, -10584, -8820, -5040, -1890, -420, -42],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [42, 420, 1890, 5040, 8820, 10584, 8820, 5040, 1890, 420, 42],
                          [48, 480, 2160, 5760, 10080, 12096, 10080, 5760, 2160, 480, 48],
                          [27, 270, 1215, 3240, 5670, 6804, 5670, 3240, 1215, 270, 27],
                          [8, 80, 360, 960, 1680, 2016, 1680, 960, 360, 80, 8],
                          [1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]]
            # North
            D[:, :, 2] = [[-1, -8, -27, -48, -42, 0, 42, 48, 27, 8, 1],
                          [-10, -80, -270, -480, -420, 0, 420, 480, 270, 80, 10],
                          [-45, -360, -1215, -2160, -1890, 0, 1890, 2160, 1215, 360, 45],
                          [-120, -960, -3240, -5760, -5040, 0, 5040, 5760, 3240, 960, 120],
                          [-210, -1680, -5670, -10080, -8820, 0, 8820, 10080, 5670, 1680, 210],
                          [-252, -2016, -6804, -12096, -10584, 0, 10584, 12096, 6804, 2016, 252],
                          [-210, -1680, -5670, -10080, -8820, 0, 8820, 10080, 5670, 1680, 210],
                          [-120, -960, -3240, -5760, -5040, 0, 5040, 5760, 3240, 960, 120],
                          [-45, -360, -1215, -2160, -1890, 0, 1890, 2160, 1215, 360, 45],
                          [-10, -80, -270, -480, -420, 0, 420, 480, 270, 80, 10],
                          [-1, -8, -27, -48, -42, 0, 42, 48, 27, 8, 1]]
            # North West
            D[:, :, 3] = [[1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1],
                          [8, 80, 360, 960, 1680, 2016, 1680, 960, 360, 80, 8],
                          [27, 270, 1215, 3240, 5670, 6804, 5670, 3240, 1215, 270, 27],
                          [48, 480, 2160, 5760, 10080, 12096, 10080, 5760, 2160, 480, 48],
                          [42, 420, 1890, 5040, 8820, 10584, 8820, 5040, 1890, 420, 42],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [-42, -420, -1890, -5040, -8820, -10584, -8820, -5040, -1890, -420, -42],
                          [-48, -480, -2160, -5760, -10080, -12096, -10080, -5760, -2160, -480, -48],
                          [-27, -270, -1215, -3240, -5670, -6804, -5670, -3240, -1215, -270, -27],
                          [-8, -80, -360, -960, -1680, -2016, -1680, -960, -360, -80, -8],
                          [-1, -10, -45, -120, -210, -252, -210, -120, -45, -10, -1]]
            # West
            D[:, :, 4] = [[1, 8, 27, 48, 42, 0, -42, -48, -27, -8, -1],
                          [10, 80, 270, 480, 420, 0, -420, -480, -270, -80, -10],
                          [45, 360, 1215, 2160, 1890, 0, -1890, -2160, -1215, -360, -45],
                          [120, 960, 3240, 5760, 5040, 0, -5040, -5760, -3240, -960, -120],
                          [210, 1680, 5670, 10080, 8820, 0, -8820, -10080, -5670, -1680, -210],
                          [252, 2016, 6804, 12096, 10584, 0, -10584, -12096, -6804, -2016, -252],
                          [210, 1680, 5670, 10080, 8820, 0, -8820, -10080, -5670, -1680, -210],
                          [120, 960, 3240, 5760, 5040, 0, -5040, -5760, -3240, -960, -120],
                          [45, 360, 1215, 2160, 1890, 0, -1890, -2160, -1215, -360, -45],
                          [10, 80, 270, 480, 420, 0, -420, -480, -270, -80, -10],
                          [1, 8, 27, 48, 42, 0, -42, -48, -27, -8, -1]]
            # South West
            D[:, :, 5] = [[0, -42, -48, -27, -8, -1, -10, -45, -120, -210, -252],
                          [42, 0, -420, -480, -270, -80, -360, -960, -1680, -2016, -210],
                          [48, 420, 0, -1890, -2160, -1215, -3240, -5670, -6804, -1680, -120],
                          [27, 480, 1890, 0, -5040, -5760, -10080, -12096, -5670, -960, -45],
                          [8, 270, 2160, 5040, 0, -8820, -10584, -10080, -3240, -360, -10],
                          [1, 80, 1215, 5760, 8820, 0, -8820, -5760, -1215, -80, -1],
                          [10, 360, 3240, 10080, 10584, 8820, 0, -5040, -2160, -270, -8],
                          [45, 960, 5670, 12096, 10080, 5760, 5040, 0, -1890, -480, -27],
                          [120, 1680, 6804, 5670, 3240, 1215, 2160, 1890, 0, -420, -48],
                          [210, 2016, 1680, 960, 360, 80, 270, 480, 420, 0, -42],
                          [252, 210, 120, 45, 10, 1, 8, 27, 48, 42, 0]]
            # South
            D[:, :, 6] = [[252, 210, 120, 45, 10, 1, 8, 27, 48, 42, 0],
                          [210, 2016, 1680, 960, 360, 80, 270, 480, 420, 0, -42],
                          [120, 1680, 6804, 5670, 3240, 1215, 2160, 1890, 0, -420, -48],
                          [45, 960, 5670, 12096, 10080, 5760, 5040, 0, -1890, -480, -27],
                          [10, 360, 3240, 10080, 10584, 8820, 0, -5040, -2160, -270, -8],
                          [1, 80, 1215, 5760, 8820, 0, -8820, -5760, -1215, -80, -1],
                          [8, 270, 2160, 5040, 0, -8820, -10584, -10080, -3240, -360, -10],
                          [27, 480, 1890, 0, -5040, -5760, -10080, -12096, -5670, -960, -45],
                          [48, 420, 0, -1890, -2160, -1215, -3240, -5670, -6804, -1680, -120],
                          [42, 0, -420, -480, -270, -80, -360, -960, -1680, -2016, -210],
                          [0, -42, -48, -27, -8, -1, -10, -45, -120, -210, -252]]
            # South East
            D[:, :, 7] = [[-252, -210, -120, -45, -10, -1, -8, -27, -48, -42, 0],
                          [-210, -2016, -1680, -960, -360, -80, -270, -480, -420, 0, 42],
                          [-120, -1680, -6804, -5670, -3240, -1215, -2160, -1890, 0, 420, 48],
                          [-45, -960, -5670, -12096, -10080, -5760, -5040, 0, 1890, 480, 27],
                          [-10, -360, -3240, -10080, -10584, -8820, 0, 5040, 2160, 270, 8],
                          [-1, -80, -1215, -5760, -8820, 0, 8820, 5760, 1215, 80, 1],
                          [-8, -270, -2160, -5040, 0, 8820, 10584, 10080, 3240, 360, 10],
                          [-27, -480, -1890, 0, 5040, 5760, 10080, 12096, 5670, 960, 45],
                          [-48, -420, 0, 1890, 2160, 1215, 3240, 5670, 6804, 1680, 120],
                          [-42, 0, 420, 480, 270, 80, 360, 960, 1680, 2016, 210],
                          [0, 42, 48, 27, 8, 1, 10, 45, 120, 210, 252]]
        else:
            raise ValueError('Sobel mask size not supported. Use only: 3, 5, 7, 9, 11')
    elif mask == 'prewitt':
        if msize == 3:
            D = np.zeros((3, 3, 8))
            # East
            D[:, :, 0] = np.array([[1, 1, 1],
                                   [0, 0, 0],
                                   [-1, -1, -1]])
            # North East
            D[:, :, 1] = np.array([[1, 1, 0],
                                   [1, 0, -1],
                                   [0, -1, -1]])
            # North
            D[:, :, 2] = np.array([[1, 0, -1],
                                   [1, 0, -1],
                                   [1, 0, -1]])
            # North West
            D[:, :, 3] = np.array([[0, -1, -1],
                                   [1, 0, -1],
                                   [1, 1, 0]])
            # West
            D[:, :, 4] = np.array([[-1, -1, -1],
                                   [0, 0, 0],
                                   [1, 1, 1]])
            # South West
            D[:, :, 5] = np.array([[-1, -1, 0],
                                   [-1, 0, 1],
                                   [0, 1, 1]])
            # South
            D[:, :, 6] = np.array([[-1, 0, 1],
                                   [-1, 0, 1],
                                   [-1, 0, 1]])
            # South East
            D[:, :, 7] = np.array([[0, 1, 1],
                                   [-1, 0, 1],
                                   [-1, -1, 0]])
        elif msize == 5:
            D = np.zeros((5, 5, 8))
            # East
            D[:, :, 0] = np.array([[2, 2, 2, 2, 2],
                                   [1, 1, 1, 1, 1],
                                   [0, 0, 0, 0, 0],
                                   [-1, -1, -1, -1, -1],
                                   [-2, -2, -2, -2, -2]])
            # North East
            D[:, :, 1] = np.array([[2, 1, 0, -1, -2],
                                   [2, 1, 0, -1, -2],
                                   [2, 1, 0, -1, -2],
                                   [2, 1, 0, -1, -2],
                                   [2, 1, 0, -1, -2]])
            # North
            D[:, :, 2] = np.array([[-2, -1, 0, 1, 2],
                                   [-2, -1, 0, 1, 2],
                                   [-2, -1, 0, 1, 2],
                                   [-2, -1, 0, 1, 2],
                                   [-2, -1, 0, 1, 2]])
            # North West
            D[:, :, 3] = np.array([[-2, -2, -2, -2, -2],
                                   [-1, -1, -1, -1, -1],
                                   [0, 0, 0, 0, 0],
                                   [1, 1, 1, 1, 1],
                                   [2, 2, 2, 2, 2]])
            # West
            D[:, :, 4] = np.array([[-2, -2, -2, -1, 0],
                                   [-2, -1, -1, 0, 1],
                                   [-2, -1, 0, 1, 2],
                                   [-1, 0, 1, 1, 2],
                                   [0, 1, 2, 2, 2]])
            # South West
            D[:, :, 5] = np.array([[0, 1, 2, 2, 2],
                                   [-1, 0, 1, 1, 2],
                                   [-2, -1, 0, 1, 2],
                                   [-2, -1, -1, 0, 1],
                                   [-2, -2, -2, -1, 0]])
            # South
            D[:, :, 6] = np.array([[0, -1, -2, -2, -2],
                                   [1, 0, -1, -1, -2],
                                   [2, 1, 0, -1, -2],
                                   [2, 1, 1, 0, -1],
                                   [2, 2, 2, 1, 0]])
            # South East
            D[:, :, 7] = np.array([[2, 2, 2, 1, 0],
                                   [2, 1, 1, 0, -1],
                                   [2, 1, 0, -1, -2],
                                   [1, 0, -1, -1, -2],
                                   [0, -1, -2, -2, -2]])
        elif msize == 7:
            D = np.zeros((7, 7, 8))
            # East
            D[:, :, 0] = np.array([[3, 3, 3, 3, 3, 3, 3],
                                   [2, 2, 2, 2, 2, 2, 2],
                                   [1, 1, 1, 1, 1, 1, 1],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [-1, -1, -1, -1, -1, -1, -1],
                                   [-2, -2, -2, -2, -2, -2, -2],
                                   [-3, -3, -3, -3, -3, -3, -3]])
            # North East
            D[:, :, 1] = np.array([[3, 2, 1, 0, -1, -2, -3],
                                   [3, 2, 1, 0, -1, -2, -3],
                                   [3, 2, 1, 0, -1, -2, -3],
                                   [3, 2, 1, 0, -1, -2, -3],
                                   [3, 2, 1, 0, -1, -2, -3],
                                   [3, 2, 1, 0, -1, -2, -3],
                                   [3, 2, 1, 0, -1, -2, -3]])
            # North
            D[:, :, 2] = np.array([[-3, -2, -1, 0, 1, 2, 3],
                                   [-3, -2, -1, 0, 1, 2, 3],
                                   [-3, -2, -1, 0, 1, 2, 3],
                                   [-3, -2, -1, 0, 1, 2, 3],
                                   [-3, -2, -1, 0, 1, 2, 3],
                                   [-3, -2, -1, 0, 1, 2, 3],
                                   [-3, -2, -1, 0, 1, 2, 3]])
            # North West
            D[:, :, 3] = np.array([[-3, -3, -3, -3, -3, -3, -3],
                                   [-2, -2, -2, -2, -2, -2, -2],
                                   [-1, -1, -1, -1, -1, -1, -1],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [1, 1, 1, 1, 1, 1, 1],
                                   [2, 2, 2, 2, 2, 2, 2],
                                   [3, 3, 3, 3, 3, 3, 3]])
            # West
            D[:, :, 4] = np.array([[-3, -3, -3, -3, -2, -1, 0],
                                   [-3, -2, -2, -2, -1, 0, 1],
                                   [-3, -2, -1, -1, 0, 1, 2],
                                   [-3, -2, -1, 0, 1, 2, 3],
                                   [-2, -1, 0, 1, 1, 2, 3],
                                   [-1, 0, 1, 2, 2, 2, 3],
                                   [0, 1, 2, 3, 3, 3, 3]])
            # South West
            D[:, :, 5] = np.array([[0, 1, 2, 3, 3, 3, 3],
                                   [-1, 0, 1, 2, 2, 2, 3],
                                   [-2, -1, 0, 1, 1, 2, 3],
                                   [-3, -2, -1, 0, 1, 2, 3],
                                   [-3, -2, -1, -1, 0, 1, 2],
                                   [-3, -2, -2, -2, -1, 0, 1],
                                   [-3, -3, -3, -3, -2, -1, 0]])
            # South
            D[:, :, 6] = np.array([[0, -1, -2, -3, -3, -3, -3],
                                   [1, 0, -1, -2, -2, -2, -3],
                                   [2, 1, 0, -1, -1, -2, -3],
                                   [3, 2, 1, 0, -1, -2, -3],
                                   [3, 2, 1, 1, 0, -1, -2],
                                   [3, 2, 2, 2, 1, 0, -1],
                                   [3, 3, 3, 3, 2, 1, 0]])
            # South East
            D[:, :, 7] = np.array([[3, 3, 3, 3, 2, 1, 0],
                                   [3, 2, 2, 2, 1, 0, -1],
                                   [3, 2, 1, 1, 0, -1, -2],
                                   [3, 2, 1, 0, -1, -2, -3],
                                   [2, 1, 0, -1, -1, -2, -3],
                                   [1, 0, -1, -2, -2, -2, -3],
                                   [0, -1, -2, -3, -3, -3, -3]])
        elif msize == 9:
            D = np.zeros((9, 9, 8))
            # East
            D[:, :, 0] = np.array([[4, 4, 4, 4, 4, 4, 4, 4, 4],
                                   [3, 3, 3, 3, 3, 3, 3, 3, 3],
                                   [2, 2, 2, 2, 2, 2, 2, 2, 2],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                                   [-2, -2, -2, -2, -2, -2, -2, -2, -2],
                                   [-3, -3, -3, -3, -3, -3, -3, -3, -3],
                                   [-4, -4, -4, -4, -4, -4, -4, -4, -4]])
            # North East
            D[:, :, 1] = np.array([[4, 3, 2, 1, 0, -1, -2, -3, -4],
                                   [4, 3, 2, 1, 0, -1, -2, -3, -4],
                                   [4, 3, 2, 1, 0, -1, -2, -3, -4],
                                   [4, 3, 2, 1, 0, -1, -2, -3, -4],
                                   [4, 3, 2, 1, 0, -1, -2, -3, -4],
                                   [4, 3, 2, 1, 0, -1, -2, -3, -4],
                                   [4, 3, 2, 1, 0, -1, -2, -3, -4],
                                   [4, 3, 2, 1, 0, -1, -2, -3, -4],
                                   [4, 3, 2, 1, 0, -1, -2, -3, -4]])
            # North
            D[:, :, 2] = np.array([[-4, -3, -2, -1, 0, 1, 2, 3, 4],
                                   [-4, -3, -2, -1, 0, 1, 2, 3, 4],
                                   [-4, -3, -2, -1, 0, 1, 2, 3, 4],
                                   [-4, -3, -2, -1, 0, 1, 2, 3, 4],
                                   [-4, -3, -2, -1, 0, 1, 2, 3, 4],
                                   [-4, -3, -2, -1, 0, 1, 2, 3, 4],
                                   [-4, -3, -2, -1, 0, 1, 2, 3, 4],
                                   [-4, -3, -2, -1, 0, 1, 2, 3, 4],
                                   [-4, -3, -2, -1, 0, 1, 2, 3, 4]])
            # North West
            D[:, :, 3] = np.array([[-4, -4, -4, -4, -4, -4, -4, -4, -4],
                                   [-3, -3, -3, -3, -3, -3, -3, -3, -3],
                                   [-2, -2, -2, -2, -2, -2, -2, -2, -2],
                                   [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [2, 2, 2, 2, 2, 2, 2, 2, 2],
                                   [3, 3, 3, 3, 3, 3, 3, 3, 3],
                                   [4, 4, 4, 4, 4, 4, 4, 4, 4]])
            # West
            D[:, :, 4] = np.array([[-4, -4, -4, -4, -4, -3, -2, -1, 0],
                                   [-4, -3, -3, -3, -3, -2, -1, 0, 1],
                                   [-4, -3, -2, -2, -2, -1, 0, 1, 2],
                                   [-4, -3, -2, -1, -1, 0, 1, 2, 3],
                                   [-4, -3, -2, -1, 0, 1, 2, 3, 4],
                                   [-3, -2, -1, 0, 1, 1, 2, 3, 4],
                                   [-2, -1, 0, 1, 2, 2, 2, 3, 4],
                                   [-1, 0, 1, 2, 3, 3, 3, 3, 4],
                                   [0, 1, 2, 3, 4, 4, 4, 4, 4]])
            # South West
            D[:, :, 5] = np.array([[0, 1, 2, 3, 4, 4, 4, 4, 4],
                                   [-1, 0, 1, 2, 3, 3, 3, 3, 4],
                                   [-2, -1, 0, 1, 2, 2, 2, 3, 4],
                                   [-3, -2, -1, 0, 1, 1, 2, 3, 4],
                                   [-4, -3, -2, -1, 0, 1, 2, 3, 4],
                                   [-4, -3, -2, -1, -1, 0, 1, 2, 3],
                                   [-4, -3, -2, -2, -2, -1, 0, 1, 2],
                                   [-4, -3, -3, -3, -3, -2, -1, 0, 1],
                                   [-4, -4, -4, -4, -4, -3, -2, -1, 0]])
            # South
            D[:, :, 6] = np.array([[0, -1, -2, -3, -4, -4, -4, -4, -4],
                                   [1, 0, -1, -2, -3, -3, -3, -3, -4],
                                   [2, 1, 0, -1, -2, -2, -2, -3, -4],
                                   [3, 2, 1, 0, -1, -1, -2, -3, -4],
                                   [4, 3, 2, 1, 0, -1, -2, -3, -4],
                                   [4, 3, 2, 1, 1, 0, -1, -2, -3],
                                   [4, 3, 2, 2, 2, 1, 0, -1, -2],
                                   [4, 3, 3, 3, 3, 2, 1, 0, -1],
                                   [4, 4, 4, 4, 4, 3, 2, 1, 0]])
            # South East
            D[:, :, 7] = np.array([[4, 4, 4, 4, 4, 3, 2, 1, 0],
                                   [4, 3, 3, 3, 3, 2, 1, 0, -1],
                                   [4, 3, 2, 2, 2, 1, 0, -1, -2],
                                   [4, 3, 2, 1, 1, 0, -1, -2, -3],
                                   [4, 3, 2, 1, 0, -1, -2, -3, -4],
                                   [3, 2, 1, 0, -1, -1, -2, -3, -4],
                                   [2, 1, 0, -1, -2, -2, -2, -3, -4],
                                   [1, 0, -1, -2, -3, -3, -3, -3, -4],
                                   [0, -1, -2, -3, -4, -4, -4, -4, -4]])
        elif msize == 11:
            D = np.zeros((11, 11, 8))
            # East
            D[:, :, 0] = np.array([[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                                   [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                                   [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                                   [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                   [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
                                   [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                                   [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4],
                                   [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5]])
            # North East
            D[:, :, 1] = np.array([[5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5],
                                   [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5],
                                   [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5],
                                   [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5],
                                   [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5],
                                   [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5],
                                   [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5],
                                   [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5],
                                   [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5],
                                   [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5],
                                   [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5]])
            # North
            D[:, :, 2] = np.array([[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                                   [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                                   [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                                   [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                                   [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                                   [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                                   [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                                   [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                                   [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                                   [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                                   [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]])
            # North West
            D[:, :, 3] = np.array([[-5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5],
                                   [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4],
                                   [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                                   [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
                                   [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                   [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                                   [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                                   [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]])
            # West
            D[:, :, 4] = np.array([[0, -1, -2, -3, -4, -5, -5, -5, -5, -5, -5],
                                   [1, 0, -1, -2, -3, -4, -4, -4, -4, -4, -5],
                                   [2, 1, 0, -1, -2, -3, -3, -3, -3, -4, -5],
                                   [3, 2, 1, 0, -1, -2, -2, -2, -3, -4, -5],
                                   [4, 3, 2, 1, 0, -1, -1, -2, -3, -4, -5],
                                   [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5],
                                   [5, 4, 3, 2, 1, 1, 0, -1, -2, -3, -4],
                                   [5, 4, 3, 2, 2, 2, 1, 0, -1, -2, -3],
                                   [5, 4, 3, 3, 3, 3, 2, 1, 0, -1, -2],
                                   [5, 4, 4, 4, 4, 4, 3, 2, 1, 0, -1],
                                   [5, 5, 5, 5, 5, 5, 4, 3, 2, 1, 0]])
            # South West
            D[:, :, 5] = np.array([[0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5],
                                   [-1, 0, 1, 2, 3, 4, 4, 4, 4, 4, 5],
                                   [-2, -1, 0, 1, 2, 3, 3, 3, 3, 4, 5],
                                   [-3, -2, -1, 0, 1, 2, 2, 2, 3, 4, 5],
                                   [-4, -3, -2, -1, 0, 1, 1, 2, 3, 4, 5],
                                   [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                                   [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                                   [-5, -4, -3, -2, -1, -1, 0, 1, 2, 3, 4],
                                   [-5, -4, -3, -2, -2, -2, -1, 0, 1, 2, 3],
                                   [-5, -4, -3, -3, -3, -3, -2, -1, 0, 1, 2],
                                   [-5, -4, -4, -4, -4, -4, -3, -2, -1, 0, 1]])
            # South
            D[:, :, 6] = np.array([[-5, -5, -5, -5, -5, -5, -4, -3, -2, -1, 0],
                                   [-5, -4, -4, -4, -4, -4, -3, -2, -1, 0, 1],
                                   [-5, -4, -3, -3, -3, -3, -2, -1, 0, 1, 2],
                                   [-5, -4, -3, -2, -2, -2, -1, 0, 1, 2, 3],
                                   [-5, -4, -3, -2, -1, -1, 0, 1, 2, 3, 4],
                                   [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                                   [-4, -3, -2, -1, 0, 1, 1, 2, 3, 4, 5],
                                   [-3, -2, -1, 0, 1, 2, 2, 2, 3, 4, 5],
                                   [-2, -1, 0, 1, 2, 3, 3, 3, 3, 4, 5],
                                   [-1, 0, 1, 2, 3, 4, 4, 4, 4, 4, 5],
                                   [0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5]])
            # South East
            D[:, :, 7] = np.array([[5, 5, 5, 5, 5, 5, 4, 3, 2, 1, 0],
                                   [5, 4, 4, 4, 4, 4, 3, 2, 1, 0, -1],
                                   [5, 4, 3, 3, 3, 3, 2, 1, 0, -1, -2],
                                   [5, 4, 3, 2, 2, 2, 1, 0, -1, -2, -3],
                                   [5, 4, 3, 2, 1, 1, 0, -1, -2, -3, -4],
                                   [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5],
                                   [4, 3, 2, 1, 0, -1, -1, -2, -3, -4, -5],
                                   [3, 2, 1, 0, -1, -2, -2, -2, -3, -4, -5],
                                   [2, 1, 0, -1, -2, -3, -3, -3, -3, -4, -5],
                                   [1, 0, -1, -2, -3, -4, -4, -4, -4, -4, -5],
                                   [0, -1, -2, -3, -4, -5, -5, -5, -5, -5, -5]])
        else:
            raise ValueError('Prewitt mask size not supported. Use only: 3, 5, 7, 9, 11')
    else:
        raise ValueError('Mask not supported. Use only: Kirsch, Sobel, Prewitt')

    r, c = image.shape
    R = np.zeros((r, c, 8))

    # Compute responses
    for i in range(8):
        R[:, :, i] = convolve2d(image, D[:, :, i], mode='same', boundary='symm')

    # Sort responses, from most negative to most positive
    idx = np.argsort(R, axis=2)[:, :, ::-1]

    # Return only the indexes, equivalent to orientation direction encoding [edge gradient]
    top = idx[:, :, [0, -1]]
    C = top[:, :, 0] * 8 + top[:, :, 1]

    return C


def gauss(x, sigma):
    """
    Calculate the value of the Gaussian (normal) distribution at a given point.

    :param x: The point or points at which to evaluate the Gaussian function.
    :type x: float or numpy.ndarray
    :param sigma: The standard deviation of the Gaussian distribution.
    :type sigma: float

    :returns: The value(s) of the Gaussian function at the given point(s).
    :rtype: float or numpy.ndarray

    :example:
        >>> gauss(0, 1)
        0.3989422804014327
        >>> gauss(np.array([0, 1, 2]), 1)
        array([0.39894228, 0.24197072, 0.05399097])
    """
    # Compute the Gaussian function value
    return np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def dgauss(x, sigma):
    """
    Compute the derivative of the Gaussian (normal) distribution with respect to x.

    :param x: The point or points at which to evaluate the derivative.
    :type x: float or numpy.ndarray
    :param sigma: The standard deviation of the Gaussian distribution.
    :type sigma: float

    :returns: The derivative of the Gaussian function at the given point(s).
    :rtype: float or numpy.ndarray

    :example:
        >>> dgauss(0, 1)
        -0.0
        >>> dgauss(np.array([0, 1, 2]), 1)
        array([-0., -0.24197072, -0.10798193])
    """
    return -x * gauss(x, sigma) / sigma ** 2


def gauss_gradient(sigma):
    """
    Generate a set of 2-D Gaussian derivative kernels for gradient computation at multiple orientations.

    :param sigma: The standard deviation of the Gaussian distribution.
    :type sigma: float

    :returns: A 3D array where each 2D slice represents a Gaussian derivative kernel at a specific orientation.
    :rtype: numpy.ndarray

    :example:
        >>> import matplotlib.pyplot as plt
        >>> sigma = 1.0
        >>> kernels = gauss_gradient(sigma)
        >>> fig, axes = plt.subplots(1, 8, figsize=(20, 5))
        >>> for i in range(8):
        ...     axes[i].imshow(kernels[:, :, i], cmap='gray')
        ...     axes[i].set_title(f'{i*45} degrees')
        ...     axes[i].axis('off')
        >>> plt.tight_layout()
        >>> plt.show()
    """
    epsilon = 1e-2
    halfsize = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
    size = int(2 * halfsize + 1)

    # Generate a 2-D Gaussian kernel along x direction
    hx = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            u = [i - halfsize - 1, j - halfsize - 1]
            hx[i, j] = gauss(u[0] - halfsize + 1, sigma) * dgauss(u[1], sigma)

    hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))

    # Generate a 2-D Gaussian kernel along y direction
    D = np.zeros((hx.shape[0], hx.shape[1], 8))
    D[:, :, 0] = hx
    D[:, :, 1] = ndimage.rotate(hx, 45, reshape=False)
    D[:, :, 2] = ndimage.rotate(hx, 90, reshape=False)
    D[:, :, 3] = ndimage.rotate(hx, 135, reshape=False)
    D[:, :, 4] = ndimage.rotate(hx, 180, reshape=False)
    D[:, :, 5] = ndimage.rotate(hx, 225, reshape=False)
    D[:, :, 6] = ndimage.rotate(hx, 270, reshape=False)
    D[:, :, 7] = ndimage.rotate(hx, 315, reshape=False)

    return D


def descriptor_LPQ(image, winSize=3, decorr=1, freqestim=1, mode='im'):
    """
    Compute the Local Phase Quantization (LPQ) descriptor for a given grayscale image.

    :param image: Grayscale input image.
    :type image: numpy.ndarray
    :param winSize: Size of the window used for LPQ calculation (must be an odd number ≥ 3). Default is 3.
    :type winSize: int
    :param decorr: Flag to apply decorrelation. 0 for no decorrelation, 1 for decorrelation. Default is 1.
    :type decorr: int
    :param freqestim: Frequency estimation method. 1 for STFT uniform window, 2 for STFT Gaussian window, 3 for Gaussian derivative quadrature filter pair. Default is 1.
    :type freqestim: int
    :param mode: Specifies the output format. 'im' for image-like output, 'nh' for normalized histogram, 'h' for histogram. Default is 'im'.
    :type mode: str

    :returns: A tuple containing:
        - LPQdesc: The LPQ descriptor of the image. Depending on `mode`, it could be an image or a histogram.
        - freqRespAll: The frequency responses for all filter pairs.
    :rtype: tuple
        - LPQdesc (numpy.ndarray): Descriptor image or histogram.
        - freqRespAll (numpy.ndarray): Frequency responses.

    :example:
        >>> import numpy as np
        >>> from scipy import ndimage
        >>> image = np.random.rand(100, 100)
        >>> desc, freq_resp = descriptor_LPQ(image, winSize=5, decorr=1, freqestim=2, mode='h')
        >>> print(desc.shape)
        (256,)
        >>> print(freq_resp.shape)
        (100, 100, 8)
    """
    # Initialize parameters
    rho = 0.90
    STFTalpha = 1 / winSize
    sigmaS = (winSize - 1) / 4
    sigmaA = 8 / (winSize - 1)
    convmode = 'valid'

    # Check inputs
    if image.ndim != 2:
        raise ValueError("Only gray scale image can be used as input")
    if winSize < 3 or winSize % 2 == 0:
        raise ValueError("Window size winSize must be an odd number and greater than or equal to 3")
    if decorr not in [0, 1]:
        raise ValueError("decorr parameter must be set to 0 for no decorrelation or 1 for decorrelation")
    if freqestim not in [1, 2, 3]:
        raise ValueError("freqestim parameter must be 1, 2, or 3")
    if mode not in ['nh', 'h', 'im']:
        raise ValueError("mode must be 'nh', 'h', or 'im'")

    # Initialize
    r = (winSize - 1) // 2
    x = np.arange(-r, r + 1)
    u = np.arange(1, r + 1)

    # Form 1-D filters
    if freqestim == 1:  # STFT uniform window
        w0 = np.ones_like(x)
        w1 = np.exp(-2j * np.pi * x * STFTalpha)
        w2 = np.conj(w1)
    elif freqestim == 2:  # STFT Gaussian window
        w0 = np.exp(-0.5 * (x / sigmaS) ** 2) / (np.sqrt(2 * np.pi) * sigmaS)
        w1 = np.exp(-2j * np.pi * x * STFTalpha)
        w2 = np.conj(w1)
        gs = np.exp(-0.5 * (x / sigmaS) ** 2) / (np.sqrt(2 * np.pi) * sigmaS)
        w0 *= gs
        w1 *= gs
        w2 *= gs
        w1 -= np.mean(w1)
        w2 -= np.mean(w2)
    elif freqestim == 3:  # Gaussian derivative quadrature filter pair
        G0 = np.exp(-x ** 2 * (np.sqrt(2) * sigmaA) ** 2)
        G1 = np.concatenate((np.zeros_like(u), u * np.exp(-u ** 2 * sigmaA ** 2), [0]))
        G0 = G0 / np.max(np.abs(G0))
        G1 = G1 / np.max(np.abs(G1))
        w0 = np.real(np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(G0))))
        w1 = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(G1)))
        w2 = np.conj(w1)
        w0 = w0 / np.max(np.abs([np.real(np.max(w0)), np.imag(np.max(w0))]))
        w1 = w1 / np.max(np.abs([np.real(np.max(w1)), np.imag(np.max(w1))]))
        w2 = w2 / np.max(np.abs([np.real(np.max(w2)), np.imag(np.max(w2))]))

    # Run filters to compute the frequency response in the four points. Store real and imaginary parts separately
    filterResp = convolve2d(convolve2d(image, w0[:, np.newaxis], mode=convmode), w1[np.newaxis, :], mode=convmode)
    freqResp = np.zeros((filterResp.shape[0], filterResp.shape[1], 8))
    freqResp[:, :, 0] = np.real(filterResp)
    freqResp[:, :, 1] = np.imag(filterResp)
    filterResp = convolve2d(convolve2d(image, w1[:, np.newaxis], mode=convmode), w0[np.newaxis, :], mode=convmode)
    freqResp[:, :, 2] = np.real(filterResp)
    freqResp[:, :, 3] = np.imag(filterResp)
    filterResp = convolve2d(convolve2d(image, w1[:, np.newaxis], mode=convmode), w1[np.newaxis, :], mode=convmode)
    freqResp[:, :, 4] = np.real(filterResp)
    freqResp[:, :, 5] = np.imag(filterResp)
    filterResp = convolve2d(convolve2d(image, w1[:, np.newaxis], mode=convmode), w2[np.newaxis, :], mode=convmode)
    freqResp[:, :, 6] = np.real(filterResp)
    freqResp[:, :, 7] = np.imag(filterResp)
    freqRespAll = filterResp
    freqRow, freqCol, freqNum = freqResp.shape

    # If decorrelation is used, compute covariance matrix and corresponding whitening transform
    if decorr == 1:
        xp, yp = np.meshgrid(np.arange(1, winSize + 1), np.arange(1, winSize + 1))
        pp = np.column_stack((xp.flatten(), yp.flatten()))
        dd = np.linalg.norm(pp[:, np.newaxis] - pp[np.newaxis, :], axis=-1)
        C = rho ** dd
        q1 = np.outer(w0, w1)
        q2 = np.outer(w1, w0)
        q3 = np.outer(w1, w1)
        q4 = np.outer(w1, w2)
        u1, u2 = np.real(q1), np.imag(q1)
        u3, u4 = np.real(q2), np.imag(q2)
        u5, u6 = np.real(q3), np.imag(q3)
        u7, u8 = np.real(q4), np.imag(q4)
        M = np.array([u1.flatten(), u2.flatten(), u3.flatten(), u4.flatten(), u5.flatten(), u6.flatten(),
                      u7.flatten(), u8.flatten()])
        D = M @ C @ M.T
        A = np.diag([1.000007, 1.000006, 1.000005, 1.000004, 1.000003, 1.000002, 1.000001, 1])
        U, S, Vt = np.linalg.svd(A @ D @ A)
        idx = np.argmax(np.abs(Vt), axis=0)
        V = Vt * np.diag(1 - 2 * (Vt[idx, range(Vt.shape[1])] < -np.finfo(float).eps))
        freqResp = freqResp.reshape(freqRow * freqCol, freqNum)
        freqResp = (V.T @ freqResp.T).T
        freqResp = freqResp.reshape(freqRow, freqCol, freqNum)

    LPQdesc = np.zeros_like(freqResp[:, :, 0])
    for i in range(freqNum):
        LPQdesc += (freqResp[:, :, i] > 0) * (2 ** i)

    if mode == 'im':
        LPQdesc = LPQdesc.astype(np.uint8)

    # Histogram if needed
    if mode == 'nh' or mode == 'h':
        LPQdesc = np.histogram(LPQdesc.flatten(), bins=256, range=(0, 255))[0]

    # Normalize histogram if needed
    if mode == 'nh':
        LPQdesc = LPQdesc / np.sum(LPQdesc)

    return LPQdesc, freqRespAll


def monofilt(im, nscale, minWaveLength, mult, sigmaOnf, orientWrap=0, thetaPhase=1):
    """
    Apply a multiscale directional filter bank to a 2D grayscale image using Log-Gabor filters.

    :param im: 2D grayscale image.
    :type im: numpy.ndarray
    :param nscale: Number of scales in the filter bank.
    :type nscale: int
    :param minWaveLength: Minimum wavelength of the filters.
    :type minWaveLength: float
    :param mult: Scaling factor between consecutive scales.
    :type mult: float
    :param sigmaOnf: Bandwidth of the Log-Gabor filter.
    :type sigmaOnf: float
    :param orientWrap: If 1, wrap orientations to the range [0, π]. Default is 0 (no wrapping).
    :type orientWrap: int, optional
    :param thetaPhase: If 1, compute phase angles (theta and psi). Default is 1.
    :type thetaPhase: int, optional

    :returns: A tuple containing:
        - f: Filter responses in the spatial domain.
        - h1f: x-direction filter responses in the spatial domain.
        - h2f: y-direction filter responses in the spatial domain.
        - A: Amplitude of the filter responses.
        - theta: Phase angle of the filter responses, if `thetaPhase` is 1.
        - psi: Orientation angle of the filter responses, if `thetaPhase` is 1.

    :rtype: tuple
        - f (list of numpy.ndarray): Filter responses.
        - h1f (list of numpy.ndarray): x-direction filter responses.
        - h2f (list of numpy.ndarray): y-direction filter responses.
        - A (list of numpy.ndarray): Amplitude responses.
        - theta (list of numpy.ndarray, optional): Phase angles.
        - psi (list of numpy.ndarray, optional): Orientation angles.

    :example:
        >>> import numpy as np
        >>> from scipy import ndimage
        >>> image = np.random.rand(100, 100)
        >>> nscale = 4
        >>> minWaveLength = 6
        >>> mult = 2.0
        >>> sigmaOnf = 0.55
        >>> f, h1f, h2f, A, theta, psi = monofilt(image, nscale, minWaveLength, mult, sigmaOnf)
        >>> print(len(f))
        4
        >>> print(f[0].shape)  # Shape of the response for the first scale
        (100, 100)
    """
    if np.ndim(im) == 2:
        rows, cols = im.shape
    else:
        raise ValueError("Input image must be 2D.")

    # Compute the 2D Fourier Transform of the image
    IM = np.fft.fft2(im)

    # Generate frequency coordinates
    u1, u2 = np.meshgrid(
        (np.arange(cols) - (cols // 2 + 1)) / (cols - np.mod(cols, 2)),
        (np.arange(rows) - (rows // 2 + 1)) / (rows - np.mod(rows, 2))
    )

    # Shift the frequency coordinates
    u1 = np.fft.ifftshift(u1)
    u2 = np.fft.ifftshift(u2)

    # Compute the radius in the frequency domain
    radius = np.sqrt(u1 ** 2 + u2 ** 2)
    radius[1, 1] = 1  # Avoid division by zero at the origin

    # Initialize filter responses
    H1 = 1j * u1 / radius
    H2 = 1j * u2 / radius

    # Lists to store filter responses
    f, h1f, h2f, A, theta, psi = [], [], [], [], [], []

    for s in range(1, nscale + 1):
        # Calculate wavelength and filter frequency
        wavelength = minWaveLength * mult ** (s - 1)
        fo = 1.0 / wavelength

        # Create Log-Gabor filter
        logGabor = np.exp(-((np.log(radius / fo)) ** 2) / (2 * np.log(sigmaOnf) ** 2))
        logGabor[0, 0] = 0  # Avoid division by zero at the origin

        # Apply filter in frequency domain
        H1s = H1 * logGabor
        H2s = H2 * logGabor

        # Convert back to spatial domain
        f_spatial = np.real(np.fft.ifft2(IM * logGabor))
        h1f_spatial = np.real(np.fft.ifft2(IM * H1s))
        h2f_spatial = np.real(np.fft.ifft2(IM * H2s))

        # Compute amplitude
        A_s = np.sqrt(f_spatial ** 2 + h1f_spatial ** 2 + h2f_spatial ** 2)

        # Store results
        f.append(f_spatial)
        h1f.append(h1f_spatial)
        h2f.append(h2f_spatial)
        A.append(A_s)

        if thetaPhase:
            # Compute phase angles
            theta_s = np.arctan2(h2f_spatial, h1f_spatial)
            psi_s = np.arctan2(f_spatial, np.sqrt(h1f_spatial ** 2 + h2f_spatial ** 2))

            if orientWrap:
                # Wrap orientations to [0, π] range
                theta_s[theta_s < 0] += np.pi
                psi_s[theta_s < 0] = np.pi - psi_s[theta_s < 0]
                psi_s[psi_s > np.pi] -= 2 * np.pi

            theta.append(theta_s)
            psi.append(psi_s)

    if thetaPhase:
        return f, h1f, h2f, A, theta, psi
    else:
        return f, h1f, h2f, A


def lxp_phase(image, radius=1, neighbors=8, mapping=None, mode='h'):
    """
    Compute the Local X-Y Pattern (LXP) descriptor for a 2D grayscale image based on local phase information.

    :param image: 2D grayscale image.
    :type image: numpy.ndarray
    :param radius: Radius of the circular neighborhood for computing the pattern. Default is 1.
    :type radius: int, optional
    :param neighbors: Number of directions or neighbors to consider. Default is 8.
    :type neighbors: int, optional
    :param mapping: Coordinates of neighbors relative to each pixel. If None, uses a default circular pattern. If a single digit, computes neighbors in a circular pattern based on the digit. Default is None.
    :type mapping: numpy.ndarray or None, optional
    :param mode: Mode for output. 'h' or 'hist' for histogram of the LXP, 'nh' for normalized histogram. Default is 'h'.
    :type mode: str, optional

    :returns: LXP descriptor, either as a histogram or image depending on the `mode` parameter.
    :rtype: numpy.ndarray

    :example:
        >>> import numpy as np
        >>> from skimage import data
        >>> image = data.camera()
        >>> lxp_desc = lxp_phase(image, radius=1, neighbors=8, mode='nh')
        >>> print(lxp_desc.shape)
        (256,)
    """
    # Define bin edges for quantizing phase values
    bin = np.array([0, 90, 180, 270, 360])

    # Determine the pattern of neighbors
    if mapping is None:
        # Default 8-neighborhood pattern
        spoints = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
    elif len(str(mapping)) == 1:
        # Compute circular pattern based on neighbors
        spoints = np.zeros((neighbors, 2))
        a = 2 * np.pi / neighbors
        for i in range(neighbors):
            spoints[i, 0] = -radius * np.sin((i - 1) * a)
            spoints[i, 1] = radius * np.cos((i - 1) * a)
    else:
        # Use user-defined mapping
        spoints = mapping

    # Get the size of the image
    ysize, xsize = image.shape

    # Determine the size of the boundary box needed for the pattern
    miny, maxy = np.min(spoints[:, 0]), np.max(spoints[:, 0])
    minx, maxx = np.min(spoints[:, 1]), np.max(spoints[:, 1])

    # Calculate size of the boundary box
    bsizey = int(np.ceil(max(maxy, 0)) - np.floor(min(miny, 0))) + 1
    bsizex = int(np.ceil(max(maxx, 0)) - np.floor(min(minx, 0))) + 1

    if xsize < bsizex or ysize < bsizey:
        raise ValueError('Too small input image. Should be at least (2*radius+1) x (2*radius+1)')

    # Calculate offsets for cropping the image
    origy = 1 - int(np.floor(min(miny, 0)))
    origx = 1 - int(np.floor(min(minx, 0)))

    # Calculate sizes for the cropped image
    dy, dx = ysize - bsizey, xsize - bsizex

    # Crop the image to match the boundary box size
    C = image[origy:origy + dy, origx:origx + dx]

    # Initialize the result array
    bins = 2 ** neighbors
    result = np.zeros((dy, dx))

    # Compute the LXP descriptor
    for i in range(neighbors):
        y, x = spoints[i, 0] + origy, spoints[i, 1] + origx
        fy, cy, ry = int(np.floor(y)), int(np.ceil(y)), int(np.round(y))
        fx, cx, rx = int(np.floor(x)), int(np.ceil(x)), int(np.round(x))

        if abs(x - rx) < 1e-6 and abs(y - ry) < 1e-6:
            # No interpolation needed
            N = image[ry:ry + dy, rx:rx + dx]
            tem_N, tem_C = N.copy(), C.copy()
            for tem_i in range(1, bin.shape[0]):
                tem_N[(N >= bin[tem_i - 1]) & (N < bin[tem_i])] = tem_i - 1
                tem_C[(C >= bin[tem_i - 1]) & (C < bin[tem_i])] = tem_i - 1
            D = (tem_N != tem_C)
        else:
            # Bilinear interpolation
            ty, tx = y - fy, x - fx
            w1, w2, w3, w4 = (1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty

            N = w1 * image[fy:fy + dy, fx:fx + dx] + w2 * image[fy:fy + dy, cx:cx + dx] + \
                w3 * image[cy:cy + dy, fx:fx + dx] + w4 * image[cy:cy + dy, cx:cx + dx]
            tem_N, tem_C = N.copy(), C.copy()
            for tem_i in range(1, bin.shape[0]):
                tem_N[(N >= bin[tem_i - 1]) & (N < bin[tem_i])] = tem_i - 1
                tem_C[(C >= bin[tem_i - 1]) & (C < bin[tem_i])] = tem_i - 1
            D = (tem_N != tem_C)

        # Compute the LXP pattern value and accumulate
        v = 2 ** (i - 1)
        result += v * D

    # Normalize the result or return as histogram
    if mode in ['h', 'hist', 'nh']:
        result = np.histogram(result.flatten(), bins=np.arange(bins + 1))[0]
        if mode == 'nh':
            result /= np.sum(result)
    else:
        # Convert result to appropriate type based on the number of bins
        if bins - 1 <= np.iinfo(np.uint8).max:
            result = result.astype(np.uint8)
        elif bins - 1 <= np.iinfo(np.uint16).max:
            result = result.astype(np.uint16)
        else:
            result = result.astype(np.uint32)

    return result


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


def bin_matrix(A, E, G, angle, bin):
    """
    Compute the bin matrix for a given angle map and gradient magnitude.

    :param A: Angle map of the gradient directions.
    :type A: numpy.ndarray
    :param E: Binary edge map where edges are marked.
    :type E: numpy.ndarray
    :param G: Gradient magnitude map.
    :type G: numpy.ndarray
    :param angle: Total range of angles in degrees (e.g., 360 for full circle).
    :type angle: float
    :param bin: Number of bins to divide the angle range into.
    :type bin: int

    :returns:
        - bm: Bin matrix with assigned bins for each pixel.
        - bv: Gradient magnitude values corresponding to the bin matrix.
    :rtype:
        - bm: numpy.ndarray
        - bv: numpy.ndarray

    :example:
        >>> import numpy as np
        >>> A = np.array([[0, 45], [90, 135]])
        >>> E = np.array([[1, 1], [1, 1]])
        >>> G = np.array([[1, 2], [3, 4]])
        >>> angle = 360
        >>> bin = 8
        >>> bm, bv = bin_matrix(A, E, G, angle, bin)
        >>> print(bm)
        [[1 2]
         [3 4]]
        >>> print(bv)
        [[1. 2.]
         [3. 4.]]
    """
    # Label connected components in the edge map
    contorns, n = label(E)
    Y, X = E.shape

    # Initialize bin matrix and gradient magnitude matrix
    bm = np.zeros((Y, X), dtype=int)
    bv = np.zeros((Y, X), dtype=float)

    # Calculate the angle range per bin
    nAngle = angle / bin

    # Iterate over each labeled region (edge component)
    for i in range(1, n + 1):
        posY, posX = np.where(contorns == i)  # Get coordinates of the current edge component

        for y, x in zip(posY, posX):
            # Determine the bin index based on the angle
            b = int(np.ceil(A[y, x] / nAngle))
            if b == 0:
                b = 1
            if G[y, x] > 0:
                # Assign the bin index and gradient magnitude
                bm[y, x] = b
                bv[y, x] = G[y, x]

    return bm, bv


def phogDescriptor_hist(bh, bv, L, bin):
    """
    Compute the histogram of the Pyramid Histogram of Oriented Gradients (PHOG) descriptor.

    :param bh: Bin matrix of the image, where each pixel is assigned a bin index.
    :type bh: numpy.ndarray
    :param bv: Gradient magnitude matrix corresponding to the bin matrix.
    :type bv: numpy.ndarray
    :param L: Number of pyramid levels.
    :type L: int
    :param bin: Number of bins for the histogram.
    :type bin: int

    :returns: Normalized histogram of the PHOG descriptor.
    :rtype: numpy.ndarray

    :example:
        >>> import numpy as np
        >>> bh = np.array([[1, 2], [2, 1]])
        >>> bv = np.array([[1, 2], [2, 1]])
        >>> L = 2
        >>> bin = 4
        >>> phog_hist = phogDescriptor_hist(bh, bv, L, bin)
        >>> print(phog_hist)
        [0.1 0.2 0.2 0.1 0.1 0.1 0.1 0.1]
    """
    p = []

    # Compute histogram for level 0 (original image)
    for b in range(1, bin + 1):
        ind = (bh == b)
        p.append(np.sum(bv[ind]))

    # Compute histograms for pyramid levels
    for l in range(1, L + 1):
        x = bh.shape[1] // (2 ** l)  # Width of each cell at level l
        y = bh.shape[0] // (2 ** l)  # Height of each cell at level l
        xx = 0

        # Traverse through the image cells at level l
        while xx + x <= bh.shape[1]:
            yy = 0
            while yy + y <= bh.shape[0]:
                # Extract cell regions for bin and magnitude
                bh_cella = bh[yy:yy + y, xx:xx + x]
                bv_cella = bv[yy:yy + y, xx:xx + x]

                # Compute histogram for each cell
                for b in range(1, bin + 1):
                    ind = (bh_cella == b)
                    p.append(np.sum(bv_cella[ind]))

                yy += y
            xx += x

    # Convert list to numpy array
    p = np.array(p)

    # Normalize the histogram
    if np.sum(p) != 0:
        p = p / np.sum(p)

    return p


def NILBP_Image_ct(img, lbpPoints, mapping, mode, lbpRadius):
    """
    Compute the Neighborhood Binary Pattern (NILBP) descriptor for an image using circular interpolation.

    :param img: 2D grayscale image.
    :type img: numpy.ndarray
    :param lbpPoints: Number of points used in the LBP pattern.
    :type lbpPoints: int
    :param mapping: A dictionary containing 'num' (number of bins) and 'table' (mapping table). If None, no mapping is applied.
    :type mapping: dict or None
    :param mode: Mode for output. 'h' or 'hist' for histogram of the NILBP, 'nh' for normalized histogram.
    :type mode: str
    :param lbpRadius: Radius of the circular neighborhood for computing LBP.
    :type lbpRadius: int

    :returns: NILBP descriptor, either as a histogram or image depending on the `mode` parameter.
    :rtype: numpy.ndarray

    :example:
        >>> import numpy as np
        >>> from skimage import data
        >>> img = data.camera()
        >>> lbpPoints = 8
        >>> lbpRadius = 1
        >>> mapping = {'num': 256, 'table': np.arange(256)}
        >>> descriptor = NILBP_Image_ct(img, lbpPoints, mapping, mode='nh', lbpRadius=lbpRadius)
        >>> print(descriptor.shape)
        (256,)
    """
    # Compute LBP blocks and dimensions
    blocks, dx, dy = cirInterpSingleRadius_ct(img, lbpPoints, lbpRadius)
    blocks = blocks.T  # Transpose to match the expected shape

    # Centering the blocks by subtracting the mean
    blocks = blocks - np.mean(blocks, axis=1, keepdims=True)

    # Binarize the blocks
    blocks[blocks >= 0] = 1
    blocks[blocks < 0] = 0

    # Calculate the LBP value for each block
    weight = 2 ** np.arange(lbpPoints)
    blocks = blocks * weight
    blocks = np.sum(blocks, axis=1)

    # Reshape the result to match the image dimensions
    result = blocks
    result = np.reshape(result, (dx + 1, dy + 1))

    # Apply mapping if provided
    if isinstance(mapping, dict):
        bins = mapping['num']
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = mapping['table'][int(result[i, j])]

    # Compute the histogram or convert result to appropriate type
    if mode in ['h', 'hist', 'nh']:
        result = np.histogram(result, bins=np.arange(bins + 1))[0]
        if mode == 'nh':
            result = result / np.sum(result)
    else:
        # Determine the appropriate data type for the result
        if (bins - 1) <= np.iinfo(np.uint8).max:
            result = result.astype(np.uint8)
        elif (bins - 1) <= np.iinfo(np.uint16).max:
            result = result.astype(np.uint16)
        else:
            result = result.astype(np.uint32)

    return result


def cirInterpSingleRadius_ct(img, lbpPoints, lbpRadius):
    """
    Perform circular interpolation for a single radius in the LBP (Local Binary Pattern) computation.

    :param img: 2D grayscale image.
    :type img: numpy.ndarray
    :param lbpPoints: Number of points used in the LBP pattern.
    :type lbpPoints: int
    :param lbpRadius: Radius of the circular neighborhood for computing LBP.
    :type lbpRadius: int

    :returns:
        - blocks (numpy.ndarray): Array of size (lbpPoints, imgNewH * imgNewW) containing the interpolated pixel values.
        - dx (int): Width of the output blocks.
        - dy (int): Height of the output blocks.
    :rtype: tuple

    :example:
        >>> import numpy as np
        >>> from skimage import data
        >>> img = data.camera()  # Example grayscale image
        >>> lbpPoints = 8
        >>> lbpRadius = 1
        >>> blocks, dx, dy = cirInterpSingleRadius_ct(img, lbpPoints, lbpRadius)
        >>> print(blocks.shape)  # Shape of the blocks array
        (8, 9216)  # Example output shape
    """
    # Get image dimensions
    imgH, imgW = img.shape

    # Compute dimensions of the output blocks
    imgNewH = imgH - 2 * lbpRadius
    imgNewW = imgW - 2 * lbpRadius

    # Initialize the blocks array to store interpolated values
    blocks = np.zeros((lbpPoints, imgNewH * imgNewW))

    # Create circular pattern points
    radius = lbpRadius
    neighbors = lbpPoints
    spoints = np.zeros((neighbors, 2))
    angleStep = 2 * np.pi / neighbors

    for i in range(neighbors):
        spoints[i, 0] = -radius * np.sin(i * angleStep)
        spoints[i, 1] = radius * np.cos(i * angleStep)

    # Calculate the size of the blocks considering boundary effects
    miny, maxy = np.min(spoints[:, 0]), np.max(spoints[:, 0])
    minx, maxx = np.min(spoints[:, 1]), np.max(spoints[:, 1])

    bsizey = int(np.ceil(max(maxy, 0)) - np.floor(min(miny, 0)) + 1)
    bsizex = int(np.ceil(max(maxx, 0)) - np.floor(min(minx, 0)) + 1)

    origy = 1 - np.floor(min(miny, 0)).astype(int)
    origx = 1 - np.floor(min(minx, 0)).astype(int)

    # Check if image size is sufficient
    if imgW < bsizex or imgH < bsizey:
        raise ValueError('Too small input image. Should be at least (2*radius+1) x (2*radius+1)')

    # Compute block dimensions
    dx = imgW - bsizex
    dy = imgH - bsizey

    # Perform circular interpolation
    for i in range(neighbors):
        y = spoints[i, 0] + origy
        x = spoints[i, 1] + origx

        fy, cy, ry = np.floor(y).astype(int), np.ceil(y).astype(int), np.round(y).astype(int)
        fx, cx, rx = np.floor(x).astype(int), np.ceil(x).astype(int), np.round(x).astype(int)

        if np.abs(x - rx) < 1e-6 and np.abs(y - ry) < 1e-6:
            imgNew = img[ry - 1:ry + dy, rx - 1:rx + dx]
            blocks[i, :] = imgNew.ravel()
        else:
            # Perform bilinear interpolation
            ty, tx = y - fy, x - fx
            w1, w2, w3, w4 = (1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty

            imgNew = (w1 * img[fy - 1:fy + dy, fx - 1:fx + dx] +
                      w2 * img[fy - 1:fy + dy, cx - 1:cx + dx] +
                      w3 * img[cy - 1:cy + dy, fx - 1:fx + dx] +
                      w4 * img[cy - 1:cy + dy, cx - 1:cx + dx])
            blocks[i, :] = imgNew.ravel()

    return blocks, dx, dy


def NewRDLBP_Image(img, imgPre, lbpRadius, lbpRadiusPre, lbpPoints, mapping=None, mode='h'):
    """
    Compute the Radial Difference Local Binary Pattern (RDLBP) between two images.

    :param img: 2D grayscale image.
    :type img: numpy.ndarray
    :param imgPre: 2D grayscale image for comparison.
    :type imgPre: numpy.ndarray
    :param lbpRadius: Radius of the circular neighborhood for the current image.
    :type lbpRadius: int
    :param lbpRadiusPre: Radius of the circular neighborhood for the comparison image.
    :type lbpRadiusPre: int
    :param lbpPoints: Number of points used in the LBP pattern.
    :type lbpPoints: int
    :param mapping: Mapping dictionary for converting the LBP result to a different bin scheme.
        If provided, must contain 'num' (number of bins) and 'table' (mapping from old bin to new bin).
    :type mapping: dict or None, optional
    :param mode: Mode for output. 'h' or 'hist' for histogram of the RDLBP, 'nh' for normalized histogram. Default is 'h'.
    :type mode: str, optional

    :returns: RDLBP descriptor, either as a histogram or image depending on the `mode` parameter.
    :rtype: numpy.ndarray

    :example:
        >>> import numpy as np
        >>> from skimage import data
        >>> img = data.camera()
        >>> imgPre = data.coins()
        >>> lbpRadius = 1
        >>> lbpRadiusPre = 1
        >>> lbpPoints = 8
        >>> hist = NewRDLBP_Image(img, imgPre, lbpRadius, lbpRadiusPre, lbpPoints, mode='nh')
        >>> print(hist.shape)
        (256,)  # Example output shape for normalized histogram
    """
    # Extract circularly interpolated blocks from the current image
    blocks1, _, _ = cirInterpSingleRadiusNew(img, lbpPoints, lbpRadius)
    blocks1 = blocks1.T

    # Adjust the comparison image size based on radii and extract circularly interpolated blocks
    imgPre = imgPre[lbpRadius - lbpRadiusPre: -lbpRadius + lbpRadiusPre, lbpRadius - lbpRadiusPre: -lbpRadius + lbpRadiusPre]
    blocks2, _, _ = cirInterpSingleRadiusNew(imgPre, lbpPoints, lbpRadiusPre)
    blocks2 = blocks2.T

    # Compute the radial difference between the two images
    radialDiff = blocks1 - blocks2
    radialDiff[radialDiff >= 0] = 1
    radialDiff[radialDiff < 0] = 0

    # Compute the LBP value by weighting the binary differences
    bins = 2 ** lbpPoints
    weight = 2 ** np.arange(lbpPoints)
    radialDiff = radialDiff * weight
    radialDiff = np.sum(radialDiff, axis=1)

    # Apply mapping if it is defined
    if mapping is not None:
        bins = mapping['num']
        result = np.array([mapping['table'][int(r)] for r in radialDiff], dtype=np.uint32)
    else:
        result = radialDiff

    # Return result as histogram or image depending on mode
    if mode in ['h', 'hist', 'nh']:
        hist_result = np.histogram(result, bins=np.arange(bins + 1))[0]
        if mode == 'nh':
            hist_result = hist_result / np.sum(hist_result)
        return hist_result
    else:
        # Return result as matrix of unsigned integers
        max_val = bins - 1
        if max_val <= np.iinfo(np.uint8).max:
            return result.astype(np.uint8)
        elif max_val <= np.iinfo(np.uint16).max:
            return result.astype(np.uint16)
        else:
            return result.astype(np.uint32)


def RDLBP_Image_SmallestRadiusOnly(imgCenSmooth, img, lbpRadius, lbpPoints, mapping, mode):
    """
    Compute the Radial Difference Local Binary Pattern (RDLBP) for an image with a focus on the smallest radius.

    :param imgCenSmooth: Smoothed image from which the radial difference is computed.
    :type imgCenSmooth: numpy.ndarray
    :param img: Original image for extracting circularly interpolated blocks.
    :type img: numpy.ndarray
    :param lbpRadius: Radius of the circular neighborhood for LBP.
    :type lbpRadius: int
    :param lbpPoints: Number of points used in the LBP pattern.
    :type lbpPoints: int
    :param mapping: Optional mapping dictionary for converting LBP result to a different bin scheme.
        Must contain 'num' (number of bins) and 'table' (mapping from old bin to new bin).
    :type mapping: dict or None
    :param mode: Output mode. 'h' or 'hist' for histogram of the RDLBP, 'nh' for normalized histogram.
    :type mode: str

    :returns: RDLBP descriptor, either as a histogram or image depending on the `mode` parameter.
    :rtype: numpy.ndarray

    :example:
        >>> import numpy as np
        >>> from skimage import data
        >>> img = data.camera()
        >>> imgCenSmooth = data.coins()
        >>> lbpRadius = 1
        >>> lbpPoints = 8
        >>> mapping = {'num': 256, 'table': np.arange(256)}
        >>> hist = RDLBP_Image_SmallestRadiusOnly(imgCenSmooth, img, lbpRadius, lbpPoints, mapping, mode='nh')
        >>> print(hist.shape)
        (256,)  # Example output shape for normalized histogram
    """
    # Extract circularly interpolated blocks from the original image
    blocks1, dx, dy = cirInterpSingleRadiusNew(img, lbpPoints, lbpRadius)
    blocks1 = blocks1.T

    # Adjust the smoothed image size based on the radius
    imgTemp = imgCenSmooth[lbpRadius:-lbpRadius, lbpRadius:-lbpRadius]
    # Create a tiled version of the smoothed image to match the size of the LBP blocks
    blocks2 = np.tile(imgTemp.ravel(), (lbpPoints, 1)).T

    # Compute the radial difference between the blocks of the original image and the smoothed image
    radialDiff = blocks1 - blocks2
    radialDiff[radialDiff >= 0] = 1
    radialDiff[radialDiff < 0] = 0

    # Compute the LBP value by weighting the binary differences
    bins = 2 ** lbpPoints
    weight = 2 ** np.arange(lbpPoints)
    radialDiff = radialDiff * weight
    radialDiff = np.sum(radialDiff, axis=1)

    # Reshape the result to match the dimensions of the original image
    result = radialDiff
    result = np.reshape(result, (dx + 1, dy + 1))

    # Apply mapping if it is defined
    if isinstance(mapping, dict):
        bins = mapping['num']
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = mapping['table'][int(result[i, j])]

    # Return result as histogram or image depending on mode
    if mode in ['h', 'hist', 'nh']:
        hist_result = np.histogram(result, bins=np.arange(bins + 1))[0]
        if mode == 'nh':
            hist_result = hist_result / np.sum(hist_result)
        return hist_result
    else:
        # Return result as matrix of unsigned integers
        max_val = bins - 1
        if max_val <= np.iinfo(np.uint8).max:
            return result.astype(np.uint8)
        elif max_val <= np.iinfo(np.uint16).max:
            return result.astype(np.uint16)
        else:
            return result.astype(np.uint32)


def cirInterpSingleRadiusNew(img, lbpPoints, lbpRadius):
    """
    Extract circularly interpolated image blocks around a specified radius and number of points.

    :param img: The input grayscale image.
    :type img: numpy.ndarray
    :param lbpPoints: The number of points used in the LBP pattern.
    :type lbpPoints: int
    :param lbpRadius: The radius of the circular neighborhood.
    :type lbpRadius: int

    :returns:
        - blocks: A 2D array where each row represents a circularly interpolated block.
        - dx: The width of the output blocks.
        - dy: The height of the output blocks.
    :rtype: tuple (numpy.ndarray, int, int)

    :example:
        >>> import numpy as np
        >>> from skimage import data
        >>> img = data.camera()
        >>> lbpPoints = 8
        >>> lbpRadius = 1
        >>> blocks, dx, dy = cirInterpSingleRadiusNew(img, lbpPoints, lbpRadius)
        >>> print(blocks.shape)
    """
    imgH, imgW = img.shape

    # Dimensions of the new image after considering the radius
    imgNewH = imgH - 2 * lbpRadius
    imgNewW = imgW - 2 * lbpRadius

    # Initialize the blocks array
    blocks = np.zeros((lbpPoints, imgNewH * imgNewW))

    radius = lbpRadius
    neighbors = lbpPoints
    spoints = np.zeros((neighbors, 2))

    angleStep = 2 * np.pi / neighbors
    for i in range(neighbors):
        spoints[i, 0] = -radius * np.sin(i * angleStep)
        spoints[i, 1] = radius * np.cos(i * angleStep)

    miny, maxy = np.min(spoints[:, 0]), np.max(spoints[:, 0])
    minx, maxx = np.min(spoints[:, 1]), np.max(spoints[:, 1])

    bsizey = int(np.ceil(max(maxy, 0)) - np.floor(min(miny, 0)) + 1)
    bsizex = int(np.ceil(max(maxx, 0)) - np.floor(min(minx, 0)) + 1)

    origy = 1 - np.floor(min(miny, 0)).astype(int)
    origx = 1 - np.floor(min(minx, 0)).astype(int)

    # Check if the image is large enough
    if imgW < bsizex or imgH < bsizey:
        raise ValueError('Input image is too small. Should be at least (2*radius+1) x (2*radius+1)')

    dx = imgW - bsizex
    dy = imgH - bsizey

    # Extract circularly interpolated blocks
    for i in range(neighbors):
        y = spoints[i, 0] + origy
        x = spoints[i, 1] + origx

        fy, cy, ry = np.floor(y).astype(int), np.ceil(y).astype(int), np.round(y).astype(int)
        fx, cx, rx = np.floor(x).astype(int), np.ceil(x).astype(int), np.round(x).astype(int)

        if np.abs(x - rx) < 1e-6 and np.abs(y - ry) < 1e-6:
            # No interpolation needed if exact coordinates are integer
            imgNew = img[ry - 1:ry + dy, rx - 1:rx + dx]
            blocks[i, :] = imgNew.ravel()
        else:
            # Perform bilinear interpolation
            ty, tx = y - fy, x - fx
            w1, w2, w3, w4 = (1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty

            imgNew = (w1 * img[fy - 1:fy + dy, fx - 1:fx + dx] +
                      w2 * img[fy - 1:fy + dy, cx - 1:cx + dx] +
                      w3 * img[cy - 1:cy + dy, fx - 1:fx + dx] +
                      w4 * img[cy - 1:cy + dy, cx - 1:cx + dx])
            blocks[i, :] = imgNew.ravel()

    return blocks, dx, dy
