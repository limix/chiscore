from numpy import load
from numpy.testing import assert_allclose

from chiscore import davies_pvalue
from chiscore._data import data_file


def test_davies_pvalue():
    with data_file("davies_pvalue.npz") as filepath:
        data = load(filepath, allow_pickle=True)

    assert_allclose(davies_pvalue(*data["args"]), data["pval"])
