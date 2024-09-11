import numpy as np


def validate_bin(options):
    if 'bin' in options:
        if isinstance(options['bin'], (int, float, complex, np.number)):
            # Extract bin value for texture difference or use default
            bin = options.get('bin', 8)
    elif 'bin' not in options:
        bin = options.get('bin', 8)
    else:
        raise ValueError(f"Invalid bin. Please enter an integer or a float.")

    return bin