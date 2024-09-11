import numpy as np


def validate_uniformLBP(options):
    if 'uniformLBP' in options:
        if isinstance(options['uniformLBP'], (int, float, complex, np.number)):
            # Extract uniformLBP value for texture difference or use default
            uniformLBP = options.get('uniformLBP', 1)
    elif 'uniformLBP' not in options:
        uniformLBP = options.get('uniformLBP', 1)
    else:
        raise ValueError(f"Invalid uniformLBP. Please enter an integer or a float.")

    return uniformLBP