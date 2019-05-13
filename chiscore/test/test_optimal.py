from numpy import load
from numpy.testing import assert_allclose

from chiscore import optimal_davies_pvalue
from chiscore._data import data_file


def test_optimal_davies_pvalue():
    with data_file("optimal_davies_pvalue.npz") as filepath:
        data = load(filepath, allow_pickle=True)

    pval = optimal_davies_pvalue(*data["args"])
    assert_allclose(pval, 0.9547608685218306)


def test_optimal_davies_pvalue_nan():
    with data_file("danilo_nan.npz") as filepath:
        data = dict(load(filepath))

    pval = optimal_davies_pvalue(
        data["qmin"],
        data["MuQ"],
        data["VarQ"],
        data["KerQ"],
        data["eigh"],
        data["vareta"],
        data["Df"],
        data["tau_rho"],
        data["rho_list"],
    )
    assert_allclose(pval, 0.39344574097360585)


def test_optimal_davies_pvalue_bound():
    with data_file("bound.npz") as filepath:
        data = dict(load(filepath, allow_pickle=True))

    pval = optimal_davies_pvalue(
        data["qmin"],
        data["MuQ"],
        data["VarQ"],
        data["KerQ"],
        data["eigh"],
        data["vareta"],
        data["Df"],
        data["tau_rho"],
        data["rho_list"],
    )
    assert_allclose(pval, 0.22029543318607503)


def test_optimal_davies_inf():
    with data_file("bound.npz") as filepath:
        data = dict(load(filepath, allow_pickle=True))

    pval = optimal_davies_pvalue(
        data["qmin"],
        data["MuQ"],
        data["VarQ"],
        data["KerQ"],
        data["eigh"],
        data["vareta"],
        data["Df"],
        data["tau_rho"],
        data["rho_list"],
        1e-30,
    )
    assert_allclose(pval, 8e-30)


def main():
    q = [1.5, 3.0]
    mu = -0.5
    var = 1.0
    kur = 3.0
    w = [10.0, 0.2, 0.1, 0.3]
    remain_var = 0.5
    df = 3.4
    trho = [5.1, 0.2]
    grid = [0.0, 0.01]
    print(optimal_davies_pvalue(q, mu, var, kur, w, remain_var, df, trho, grid))


if __name__ == "__main__":
    main()
