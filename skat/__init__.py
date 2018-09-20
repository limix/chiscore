"""
SKAT test
=========

Estimate the joint significance of test statistics derived from linear combination
of chi-squared distributions.

Functions
---------
davies_pvalue  TODO.
optimal_davies_pvalue TODO.
skat_mod_liu TODO.

References
----------
[1] Lee, Seunggeun, Michael C. Wu, and Xihong Lin. "Optimal tests for rare variant
effects in sequencing association studies." Biostatistics 13.4 (2012): 762-775.
"""

from __future__ import absolute_import

from ._testit import test
from ._davies import davies_pvalue
from ._optimal import optimal_davies_pvalue
from ._liu import skat_mod_liu
from ._data import data_file

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "test",
    "davies_pvalue",
    "optimal_davies_pvalue",
    "skat_mod_liu",
    "data_file",
]
