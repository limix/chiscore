from skat import skat_davies_pvalue, data_file
from numpy import load
from numpy.testing import assert_allclose


def test_skat_davies_pvalue():
    with data_file("skat_davies_pvalue.npz") as filepath:
        data = load(filepath)

    pval = skat_davies_pvalue(*data["args"])
    assert_allclose(pval, data["pval"])
