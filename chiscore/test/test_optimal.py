from chiscore import optimal_davies_pvalue, data_file
from numpy import load
from numpy.testing import assert_allclose


def test_optimal_davies_pvalue():
    with data_file("optimal_davies_pvalue.npz") as filepath:
        data = load(filepath)

    pval = optimal_davies_pvalue(*data["args"])
    assert_allclose(pval, data["pval"])
