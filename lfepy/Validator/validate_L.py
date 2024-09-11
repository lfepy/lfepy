import numpy as np


def validate_L(options):
    if 'L' in options:
        if isinstance(options['L'], (int, float, complex, np.number)):
            # Extract L value for texture difference or use default
            L = options.get('L', 2)
    elif 'L' not in options:
        L = options.get('L', 2)
    else:
        raise ValueError(f"Invalid L. Please enter an integer or a float.")

    return L