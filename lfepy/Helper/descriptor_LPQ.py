import numpy as np
from scipy.signal import convolve2d


def descriptor_LPQ(image, winSize=3, decorr=1, freqestim=1, mode='im'):
    """
    Compute the Local Phase Quantization (LPQ) descriptor for a given grayscale image.

    This function computes the LPQ descriptor, which captures local texture information
    by analyzing the phase of the image's frequency components. The descriptor can be
    computed using different frequency estimation methods and can be returned as an image
    or a histogram based on the specified mode.

    Args:
        image (numpy.ndarray): Grayscale input image. Must be a 2D array.
        winSize (int, optional): Size of the window used for LPQ calculation. Must be an odd number ≥ 3. Default is 3.
        decorr (int, optional): Flag to apply decorrelation. 0 for no decorrelation, 1 for decorrelation. Default is 1.
        freqestim (int, optional): Frequency estimation method.
                                    1 for STFT uniform window,
                                    2 for STFT Gaussian window,
                                    3 for Gaussian derivative quadrature filter pair. Default is 1.
        mode (str, optional): Specifies the output format.
                              'im' for image-like output,
                              'nh' for normalized histogram,
                              'h' for histogram. Default is 'im'.

    Returns:
        tuple: A tuple containing:
            LPQdesc (numpy.ndarray): The LPQ descriptor of the image. Depending on `mode`, it could be an image or a histogram.
            freqRespAll (numpy.ndarray): The frequency responses for all filter pairs.

    Raises:
        ValueError: If:
            'image' is not a 2D array.
            'winSize' is not an odd number or less than 3.
            'decorr' is not 0 or 1.
            'freqestim' is not 1, 2, or 3.
            'mode' is not one of 'nh', 'h', or 'im'.

    Example:
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