import numpy as np


def validate_t_MTP(options):
    if 't' in options:
        if isinstance(options['t'], (int, float, complex, np.number)):
            # Extract t value for texture difference or use default
            t = options.get('t', 10)
    elif 't' not in options:
        t = options.get('t', 10)
    else:
        raise ValueError(f"Invalid t. Please enter an integer or a float.")

    return t