"""
Chiscore package
================

Estimate the joint significance of test statistics derived from linear combination
of chi-squared distributions.

Functions
---------
davies_pvalue
optimal_davies_pvalue
liu_sf

References
----------
[1] Lee, Seunggeun, Michael C. Wu, and Xihong Lin. "Optimal tests for rare variant
    effects in sequencing association studies." Biostatistics 13.4 (2012): 762-775.
[2] Liu, H., Tang, Y., & Zhang, H. H. (2009). A new chi-square approximation to the
    distribution of non-negative definite quadratic forms in non-central normal
    variables. Computational Statistics & Data Analysis, 53(4), 853-856.
"""
from ._davies import davies_pvalue
from ._liu import liu_sf
from ._optimal import optimal_davies_pvalue
from ._testit import test

__version__ = "0.1.0"

__all__ = ["__version__", "davies_pvalue", "liu_sf", "optimal_davies_pvalue", "test"]
