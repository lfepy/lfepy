import numpy as np


def validate_windowSize(options):
    if 'windowSize' in options:
        if isinstance(options['windowSize'], (int, float, complex, np.number)):
            # Extract uniformLBP value for texture difference or use default
            wSz = options.get('windowSize', 5)
    elif 'windowSize' not in options:
        wSz = options.get('windowSize', 5)
    else:
        raise ValueError(f"Invalid windowSize. Please enter an integer or a float.")

    return wSz