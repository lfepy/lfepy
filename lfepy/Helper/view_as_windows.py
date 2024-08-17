import numpy as np


def view_as_windows(arr, window_shape, step=1):
    """
    Create a view of an array with sliding windows.

    This function generates a view of the input array where each element in the view is a sliding window of a specified shape. The windows are extracted with a given step size.

    Args:
        arr (numpy.ndarray): The input array from which windows will be extracted.
        window_shape (tuple): Shape of the sliding window.
        step (int or tuple, optional): Step size of the sliding window. If an integer is provided, it is applied uniformly across all dimensions. Default is 1.

    Returns:
        numpy.ndarray: A view of the array with sliding windows.

    Raises:
        ValueError: If any dimension of the window shape is larger than the corresponding dimension of the array.

    Example:
        >>> import numpy as np
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