import numpy as np


def validate_orienNum(options):
    if 'orienNum' in options:
        if isinstance(options['orienNum'], (int, float, complex, np.number)):
            # Extract orienNum value for texture difference or use default
            orienNum = options.get('orienNum', 5)
    elif 'orienNum' not in options:
        orienNum = options.get('orienNum', 8)
    else:
        raise ValueError(f"Invalid orienNum. Please enter an integer or a float.")

    return orienNum