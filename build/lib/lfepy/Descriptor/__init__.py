# Define import statement
from .BPPC import BPPC
from .GDP import GDP
from .GDP2 import GDP2
from .GLTP import GLTP
from .IWBC import IWBC
from .LAP import LAP
from .LBP import LBP
from .LDiP import LDiP
from .LDiPv import LDiPv
from .LDN import LDN
from .LDTP import LDTP
from .LFD import LFD
from .LGBPHS import LGBPHS
from .LGDiP import LGDiP
from .LGIP import LGIP
from .LGP import LGP
from .LGTrP import LGTrP
from .LMP import LMP
from .LPQ import LPQ
from .LTeP import LTeP
from .LTrP import LTrP
from .MBC import MBC
from .MBP import MBP
from .MRELBP import MRELBP
from .MTP import MTP
from .PHOG import PHOG
from .WLD import WLD

__all__ = ["BPPC", "GDP", "GDP2", "GLTP", "IWBC", "LAP", "LBP", "LDiP", "LDiPv", "LDN", "LDTP", "LFD", "LGBPHS", "LGDiP",
           "LGIP", "LGP", "LGTrP", "LMP", "LPQ", "LTeP", "LTrP", "MBC", "MBP", "MRELBP", "MTP", "PHOG", "WLD"]

# __init__.py in the main package
__version__ = '1.0.9'
__author__ = ["Dr. Prof. Khalid M. Hosny", "BSc. Mahmoud A. Mohamed", "Dr. Essa E. Almazroei"]
__license__ = 'MIT'

# Example of a package-wide configuration
DEBUG = False
