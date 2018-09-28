from chiscore import davies_pvalue, data_file
from numpy import load
from numpy.testing import assert_allclose


def test_davies_pvalue():
    with data_file("davies_pvalue.npz") as filepath:
        data = load(filepath)

    assert_allclose(davies_pvalue(*data["args"]), data["pval"])


def main():
    q = 1.5
    w = [[0.3, 5.0], [5.0, 1.5]]
    davies_pvalue(q, w)


if __name__ == "__main__":
    main()
