import numpy as np


def validate_epsi(options):
    if 'epsi' in options:
        if isinstance(options['epsi'], (int, float, complex, np.number)):
            # Extract threshold value for texture difference or use default
            epsi = options.get('epsi', 15)
    elif 'epsi' not in options:
        epsi = options.get('epsi', 15)
    else:
        raise ValueError(f"Invalid epsi. Please enter an integer or a float.")

    return epsi