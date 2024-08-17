import numpy as np


def get_mapping_mrelbp(samples, mappingtype):
    """
    Generate a mapping table for Modified Rotation and Uniform Local Binary Patterns (MRELBP) codes.

    Args:
        samples (int): The number of sampling points in the LBP.
        mappingtype (str): The type of LBP mapping. Supports various uniform, rotation invariant, and modified patterns:
            'u2'
            'LBPu2'
            'LBPVu2GMPD2'
            'ri'
            'riu2'
            'MELBPVary'
            'AELBPVary'
            'GELBPEight'
            'CLBPEight'
            'ELBPEight'
            'LBPriu2Eight'
            'MELBPEight'
            'AELBPEight'
            'MELBPEightSch1'
            'MELBPEightSch2'
            'MELBPEightSch3'
            'MELBPEightSch4'
            'MELBPEightSch5'
            'MELBPEightSch6'
            'MELBPEightSch7'
            'MELBPEightSch8'
            'MELBPEightSch9'
            'MELBPEightSch10'
            'MELBPEightSch0'
            'MELBPEightSch11'
            'MELBPEightSch1Num'
            'MELBPEightSch1Count'

    Returns:
        dict: A dictionary containing the mapping information with the following keys:
            'table' (numpy.ndarray): The mapping table.
            'samples' (int): The number of sampling points.
            'num' (int): The number of patterns in the resulting LBP code.

    Example:
        >>> get_mapping_mrelbp(8, 'u2')
        {'table': array([...]), 'samples': 8, 'num': 59}
    """
    num_all_LBPs = 2 ** samples  # Total number of possible LBPs
    table = np.arange(num_all_LBPs)
    new_max = 0
    index = 0

    # Uniform 2
    if mappingtype in ['u2', 'LBPu2', 'LBPVu2GMPD2']:
        new_max = samples * (samples - 1) + 3
        for i in range(num_all_LBPs):
            # Rotate left by 1 bit
            j = (i << 1 | i >> (samples - 1)) & ((1 << samples) - 1)
            # Count the number of transitions (0-1 and 1-0) in the binary representation
            numt = bin(i ^ j).count('1')
            if numt <= 2:
                table[i] = index
                index += 1
            else:
                table[i] = new_max - 1

    # Rotation invariant
    if mappingtype == 'ri':
        tmp_map = np.full(num_all_LBPs, -1)
        for i in range(num_all_LBPs):
            rm = i
            r = i
            for j in range(1, samples):
                # Rotate left by 1 bit
                r = (r << 1 | r >> (samples - 1)) & ((1 << samples) - 1)
                if r < rm:
                    rm = r
            if tmp_map[rm] < 0:
                tmp_map[rm] = new_max
                new_max += 1
            table[i] = tmp_map[rm]

    # Uniform and Rotation invariant
    if mappingtype in ['riu2', 'MELBPVary', 'AELBPVary', 'GELBPEight', 'CLBPEight', 'ELBPEight',
                       'LBPriu2Eight', 'MELBPEight', 'AELBPEight', 'MELBPEightSch1', 'MELBPEightSch2',
                       'MELBPEightSch3', 'MELBPEightSch4', 'MELBPEightSch5', 'MELBPEightSch6',
                       'MELBPEightSch7', 'MELBPEightSch8', 'MELBPEightSch9', 'MELBPEightSch10',
                       'MELBPEightSch0', 'MELBPEightSch11']:
        new_max = samples + 2
        for i in range(num_all_LBPs):
            # Rotate left by 1 bit
            j = (i << 1 | i >> (samples - 1)) & ((1 << samples) - 1)
            # Count the number of transitions (0-1 and 1-0) in the binary representation
            numt = bin(i ^ j).count('1')
            if numt <= 2:
                table[i] = bin(i).count('1')
            else:
                table[i] = samples + 1

    # MELBPEightSch1Num
    if mappingtype == 'MELBPEightSch1Num':
        new_max = 2 * (samples - 1)
        for i in range(num_all_LBPs):
            # Rotate left by 1 bit
            j = (i << 1 | i >> (samples - 1)) & ((1 << samples) - 1)
            # Count the number of transitions (0-1 and 1-0) in the binary representation
            numt = bin(i ^ j).count('1')
            if numt <= 2:
                table[i] = bin(i).count('1')
            else:
                num_ones_in_LBP = bin(i).count('1')
                table[i] = samples + num_ones_in_LBP - 1

    # MELBPEightSch1Count
    if mappingtype == 'MELBPEightSch1Count':
        new_max = samples + 1
        for i in range(num_all_LBPs):
            num_ones_in_LBP = bin(i).count('1')
            table[i] = num_ones_in_LBP

    # Create the mapping dictionary
    mapping = {
        'table': table,
        'samples': samples,
        'num': new_max
    }

    # If mappingtype is empty, set 'num' to total number of possible LBPs
    if mappingtype == '':
        mapping['num'] = num_all_LBPs

    return mapping