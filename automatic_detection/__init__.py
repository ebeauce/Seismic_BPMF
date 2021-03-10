# import local package modules
from .config import cfg
# import subpackages
from . import db_h5py
from . import data
from . import dataset
from . import moveouts
from . import template_search
from . import multiplet_search
from . import second_order_matched_filter
from . import clib
from . import utils
from . import plotting_utils
from . import catalog_utils

#from .automatic_detection import (
#        db_h5py, data, dataset, moveouts, template_search, multiplet_search, clib)
#
#del automatic_detection
#
#__all__
