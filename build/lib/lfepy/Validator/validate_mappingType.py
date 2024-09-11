import numpy as np
from lfepy.Helper import get_mapping


def validate_mappingType(options, radius, neighbors):
    # Validate the mask
    valid_mapping = ['full', 'ri', 'u2', 'riu2']

    # Handle mapping type and adjust bin vector accordingly
    if 'mappingType' in options and options['mappingType'] != 'full':
        mappingType = options['mappingType']
        mapping = get_mapping(neighbors, mappingType)
        if mappingType == 'u2':
            if radius == 1:
                options['binVec'] = np.arange(0, 59)
            elif radius == 2:
                options['binVec'] = np.arange(0, 243)
        elif mappingType == 'ri':
            if radius == 1:
                options['binVec'] = np.arange(0, 36)
            elif radius == 2:
                options['binVec'] = np.arange(0, 4117)
        elif mappingType == 'riu2':
            if radius == 1:
                options['binVec'] = np.arange(0, 10)
            elif radius == 2:
                options['binVec'] = np.arange(0, 16)
        else:
            raise ValueError(
                f"Invalid mapping type '{options['mappingType']}'. Valid mapping type are {valid_mapping}.")
    else:
        mapping = 0
        options['binVec'] = np.arange(0, 256)

    return options, mapping
