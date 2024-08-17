import numpy as np


def get_mapping(samples, mappingtype):
    """
    Generate a mapping table for Local Binary Patterns (LBP) codes.

    Args:
        samples (int): The number of sampling points in the LBP.
        mappingtype (str): The type of LBP mapping. Options are:
            'u2' (uniform 2)
            'ri' (rotation invariant)
            'riu2' (uniform and rotation invariant)

    Returns:
        dict: A dictionary with the following keys:
            'table' (numpy.ndarray): The mapping table.
            'samples' (int): The number of sampling points.
            'num' (int): The number of patterns in the resulting LBP code.

    Raises:
        ValueError: If an unsupported mapping type is provided.

    Example:
        >>> get_mapping(8, 'u2')
        {'table': array([...]), 'samples': 8, 'num': 59}
    """
    # Initialize the mapping table with all possible LBP codes
    table = np.arange(2 ** samples)
    newMax = 0  # Number of patterns in the resulting LBP code
    index = 0

    if mappingtype == 'u2':  # Uniform 2
        # The maximum number of uniform patterns for given samples
        newMax = samples * (samples - 1) + 3
        for i in range(2 ** samples):
            # Convert number to binary representation with fixed width
            i_bin = np.binary_repr(i, width=samples)
            # Create a circularly shifted version of the binary number
            j_bin = np.roll(list(i_bin), -1)
            # Calculate the number of 0-1 and 1-0 transitions
            numt = sum(ib != jb for ib, jb in zip(i_bin, j_bin))
            if numt <= 2:
                table[i] = index
                index += 1
            else:
                table[i] = newMax - 1

    elif mappingtype == 'ri':  # Rotation invariant
        tmpMap = np.full(2 ** samples, -1)
        for i in range(2 ** samples):
            rm = i
            r_bin = np.binary_repr(i, width=samples)
            for j in range(1, samples):
                r = int(''.join(np.roll(list(r_bin), -j)), 2)
                if r < rm:
                    rm = r
            if tmpMap[rm] < 0:
                tmpMap[rm] = newMax
                newMax += 1
            table[i] = tmpMap[rm]

    elif mappingtype == 'riu2':  # Uniform & Rotation invariant
        # The maximum number of uniform patterns for given samples in rotation-invariant setting
        newMax = samples + 2
        for i in range(2 ** samples):
            i_bin = np.binary_repr(i, width=samples)
            j_bin = np.roll(list(i_bin), -1)
            numt = sum(ib != jb for ib, jb in zip(i_bin, j_bin))
            if numt <= 2:
                table[i] = sum(int(bit) for bit in i_bin)
            else:
                table[i] = samples + 1

    else:
        raise ValueError("Unsupported mapping type. Supported types: 'u2', 'ri', 'riu2'.")

    mapping = {'table': table, 'samples': samples, 'num': newMax}
    return mapping