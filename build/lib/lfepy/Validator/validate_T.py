import numpy as np


def validate_T(options):
    if 'T' in options:
        if isinstance(options['T'], (int, float, complex, np.number)):
            # Extract T value for texture difference or use default
            T = options.get('T', 8)
    elif 'T' not in options:
        T = options.get('T', 8)
    else:
        raise ValueError(f"Invalid T. Please enter an integer or a float.")

    return T