# def test_SKAT_Optimal_PValue_Davies():
#     data = np.load("SKAT_Optimal_PValue_Davies.npz")
#     pval = SKAT_Optimal_PValue_Davies(*data["args"])
#     assert abs(pval - data["pval"]) < 1e-9


from skat import davies_pvalue, data_file
from numpy import load
from numpy.testing import assert_allclose


def test_skat_davies_pvalue():
    with data_file("skat_davies_pvalue.npz") as filepath:
        data = load(filepath)

    pval = davies_pvalue(*data["args"])["p_value"]
    assert_allclose(pval, data["pval"])
