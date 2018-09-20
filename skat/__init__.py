"""
SKAT test
=========

Functions
---------
davies_pvalue  TODO.
skat_davies_pvalue TODO.
skat_mod_liu TODO.
"""

from __future__ import absolute_import

from ._testit import test
from ._davies import davies_pvalue
from ._skat import skat_davies_pvalue
from ._liu import skat_mod_liu
from ._data import data_file

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "test",
    "davies_pvalue",
    "skat_davies_pvalue",
    "skat_mod_liu",
    "data_file",
]
