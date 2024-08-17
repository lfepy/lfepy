import pickle
from lfepy.Helper.get_mapping_mrelbp import get_mapping_mrelbp


def get_mapping_info_ct(lbp_radius, lbp_points, lbp_method):
    """
    Retrieve or generate a mapping for circular (center-symmetric) Local Binary Patterns (LBP).

    Args:
        lbp_radius (int): The radius of the LBP.
        lbp_points (int): The number of sampling points in the LBP.
        lbp_method (str): The method for LBP mapping. Options include:
            'LBPriu2'
            'MELBPVary'
            'AELBPVary'

    Returns:
        dict: A dictionary containing the mapping information with the following keys:
            'table' (numpy.ndarray): The mapping table.
            'samples' (int): The number of sampling points.
            'num' (int): The number of patterns in the resulting LBP code.

    Example:
        >>> get_mapping_info_ct(1, 24, 'LBPriu2')
        {'table': array([...]), 'samples': 24, 'num': 26}
    """
    global block_size
    block_size = lbp_radius * 2 + 1  # Calculate the block size based on radius

    mapping = None

    # Load precomputed mapping based on specific points and method
    if lbp_points == 24 and lbp_method == 'LBPriu2':
        with open('mappingLBPpoints24RIU2.pkl', 'rb') as file:
            mapping = pickle.load(file)
    elif lbp_points == 16 and lbp_method == 'LBPriu2':
        with open('mappingLBPpoints16RIU2.pkl', 'rb') as file:
            mapping = pickle.load(file)
    elif lbp_points == 16 and lbp_method == 'MELBPVary':
        with open('mappingLBPpoints16RIU2.pkl', 'rb') as file:
            mapping = pickle.load(file)
    elif lbp_points == 24 and lbp_method == 'AELBPVary':
        with open('mappingLBPpoints24RIU2.pkl', 'rb') as file:
            mapping = pickle.load(file)
    else:
        # Generate mapping dynamically if precomputed mapping is not available
        mapping = get_mapping_mrelbp(lbp_points, lbp_method)

    return mapping
