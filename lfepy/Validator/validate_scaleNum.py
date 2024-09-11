import numpy as np


def validate_scaleNum(options):
    if 'scaleNum' in options:
        if isinstance(options['scaleNum'], (int, float, complex, np.number)):
            # Extract scaleNum value for texture difference or use default
            scaleNum = options.get('scaleNum', 5)
    elif 'scaleNum' not in options:
        scaleNum = options.get('scaleNum', 5)
    else:
        raise ValueError(f"Invalid scaleNum. Please enter an integer or a float.")

    return scaleNum