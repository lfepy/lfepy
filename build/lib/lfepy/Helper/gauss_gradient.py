import numpy as np
import scipy.ndimage as ndimage
from lfepy.Helper.gauss import gauss
from lfepy.Helper.dgauss import dgauss


def gauss_gradient(sigma):
    """
    Generate a set of 2-D Gaussian derivative kernels for gradient computation at multiple orientations.

    Args:
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        numpy.ndarray: A 3D array where each 2D slice represents a Gaussian derivative kernel at a specific orientation.

    Example:
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