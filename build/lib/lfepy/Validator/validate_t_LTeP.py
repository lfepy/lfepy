import numpy as np


def validate_t_LTeP(options):
    if 't' in options:
        if isinstance(options['t'], (int, float, complex, np.number)):
            # Extract t value for texture difference or use default
            t = options.get('t', 2)
    elif 't' not in options:
        t = options.get('t', 2)
    else:
        raise ValueError(f"Invalid t. Please enter an integer or a float.")

    return t