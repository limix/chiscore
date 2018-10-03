from chiscore import optimal_davies_pvalue, data_file
from numpy import load
from numpy.testing import assert_allclose


def test_optimal_davies_pvalue():
    with data_file("optimal_davies_pvalue.npz") as filepath:
        data = load(filepath)

    pval = optimal_davies_pvalue(*data["args"])
    assert_allclose(pval, 0.9548533861981191)


def main():
    q = [1.5, 3.0]
    mu = -0.5
    var = 1.0
    kur = 3.0
    w = [10.0, 0.2, 0.1, 0.3]
    remain_var = 0.5
    df = 3.4
    trho = [5.1, 0.2]
    grid = [0., 0.01]
    print(optimal_davies_pvalue(q, mu, var, kur, w, remain_var, df, trho, grid))


if __name__ == "__main__":
    main()
