import numpy as np


def validate_msize(options):
    if 'msize' in options:
        if isinstance(options['msize'], (int, float, complex, np.number)):
            # Extract orienNum value for texture difference or use default
            msize = options.get('msize', 3)
    elif 'msize' not in options:
        msize = options.get('msize', 3)
    else:
        raise ValueError(f"Invalid msize. Please enter an integer or a float.")

    return msize