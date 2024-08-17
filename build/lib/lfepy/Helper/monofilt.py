import numpy as np


def monofilt(im, nscale, minWaveLength, mult, sigmaOnf, orientWrap=0, thetaPhase=1):
    """
    Apply a multiscale directional filter bank to a 2D grayscale image using Log-Gabor filters.

    Args:
        im (numpy.ndarray): 2D grayscale image.
        nscale (int): Number of scales in the filter bank.
        minWaveLength (float): Minimum wavelength of the filters.
        mult (float): Scaling factor between consecutive scales.
        sigmaOnf (float): Bandwidth of the Log-Gabor filter.
        orientWrap (int, optional): If 1, wrap orientations to the range [0, π]. Default is 0 (no wrapping).
        thetaPhase (int, optional): If 1, compute phase angles (theta and psi). Default is 1.

    Returns:
        tuple: A tuple containing:
            f (list of numpy.ndarray): Filter responses in the spatial domain.
            h1f (list of numpy.ndarray): x-direction filter responses in the spatial domain.
            h2f (list of numpy.ndarray): y-direction filter responses in the spatial domain.
            A (list of numpy.ndarray): Amplitude of the filter responses.
            theta (list of numpy.ndarray, optional): Phase angles of the filter responses, if `thetaPhase` is 1.
            psi (list of numpy.ndarray, optional): Orientation angles of the filter responses, if `thetaPhase` is 1.

    Raises:
        ValueError: If the input image is not 2D.

    Example:
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
        >>> print(f[0].shape)
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