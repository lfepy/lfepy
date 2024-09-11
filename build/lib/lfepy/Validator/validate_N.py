import numpy as np


def validate_N(options):
    if 'N' in options:
        if isinstance(options['N'], (int, float, complex, np.number)):
            # Extract N value for texture difference or use default
            N = options.get('N', 4)
    elif 'T' not in options:
        N = options.get('N', 4)
    else:
        raise ValueError(f"Invalid N. Please enter an integer or a float.")

    return N