__all__ = ["dataset", "template_search", "similarity_search", "utils",
           "plotting_utils", "clib"]

# import local package modules
from .config import cfg

# import subpackages
from . import dataset
from . import template_search
from . import similarity_search
from . import clib
from . import utils
from . import plotting_utils
from . import spectrum

__version__ = "2.0.0.beta"
