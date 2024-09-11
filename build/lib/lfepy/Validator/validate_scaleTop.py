import numpy as np


def validate_scaleTop(options):
    if 'scaleTop' in options:
        if isinstance(options['scaleTop'], (int, float, complex, np.number)):
            # Extract scaleTop value for texture difference or use default
            scaleTop = options.get('scaleTop', 1)
    elif 'scaleTop' not in options:
        scaleTop = options.get('scaleTop', 1)
    else:
        raise ValueError(f"Invalid scaleTop. Please enter an integer or a float.")

    return scaleTop