import numpy as np


def validate_angle(options):
    if 'angle' in options:
        if isinstance(options['angle'], (int, float, complex, np.number)):
            # Extract angle value for texture difference or use default
            angle = options.get('angle', 360)
    elif 'angle' not in options:
        angle = options.get('angle', 360)
    else:
        raise ValueError(f"Invalid angle. Please enter an integer or a float.")

    return angle