from chiscore import davies_pvalue, data_file
from numpy import load
from numpy.testing import assert_allclose


def test_davies_pvalue():
    with data_file("davies_pvalue.npz") as filepath:
        data = load(filepath)

    pval = davies_pvalue(*data["args"])["p_value"]
    assert_allclose(pval, data["pval"])
