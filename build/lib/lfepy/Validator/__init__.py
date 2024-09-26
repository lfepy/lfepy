# Define import statement
from .validate_image import validate_image
from .validate_kwargs import validate_kwargs
from .validate_mode import validate_mode
from .validate_mask_GDP import validate_mask_GDP
from .validate_DGLP import validate_DGLP
from .validate_mappingType import validate_mappingType
from .validate_radius import validate_radius
from .validate_epsi import validate_epsi
from .validate_uniformLBP import validate_uniformLBP
from .validate_scaleNum import validate_scaleNum
from .validate_orienNum import validate_orienNum
from .validate_windowSize import validate_windowSize
from .validate_t_LTeP import validate_t_LTeP
from .validate_mbcMode import validate_mbcMode
from .validate_t_MTP import validate_t_MTP
from .validate_mask_LDN import validate_mask_LDN
from .validate_msize import validate_msize
from .validate_angle import validate_angle
from .validate_bin import validate_bin
from .validate_L import validate_L
from .validate_T import validate_T
from .validate_N import validate_N
from .validate_scaleTop import validate_scaleTop


__all__ = ["validate_image", "validate_kwargs", "validate_mode", "validate_mask_GDP", "validate_DGLP",
           "validate_mappingType", "validate_radius", "validate_epsi", "validate_uniformLBP", "validate_scaleNum",
           "validate_orienNum", "validate_windowSize", "validate_t_LTeP", "validate_mbcMode", "validate_t_MTP",
           "validate_mask_LDN", "validate_msize", "validate_angle", "validate_bin", "validate_L", "validate_T",
           "validate_N", "validate_scaleTop"]

# __init__.py in the main package
__version__ = '1.0.9'
__author__ = ["Dr. Prof. Khalid M. Hosny", "BSc. Mahmoud A. Mohamed", "Dr. Essa E. Almazroei"]
__license__ = 'MIT'

# Example of a package-wide configuration
DEBUG = False
