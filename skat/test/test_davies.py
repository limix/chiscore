from skat import davies_pvalue
from numpy import load


def test_davies_pvalue():
    data = load("davies_pvalue.npz")
    pval = davies_pvalue(*data["args"])["p_value"]
    assert abs(pval - data["pval"]) < 1e-9
