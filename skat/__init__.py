"""
SKAT test
=========

Functions
---------
davies_pvalue  TODO.
"""

from __future__ import absolute_import

from ._testit import test
from ._davies import davies_pvalue
from ._data import data_file

__version__ = "0.0.1"

__all__ = ["__version__", "test", "davies_pvalue", "data_file"]

