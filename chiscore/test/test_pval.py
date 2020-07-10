from numpy import asarray, diag, median, random
from numpy.testing import assert_allclose
from scipy.stats import ncx2

from chiscore import davies_pvalue, liu_sf


def test_pval_calibration1():
    dof = [1, 1, 1]
    w = [0.5, 0.4, 0.1]
    loc = [0.0, 0.0, 0.0]

    random.seed(1)
    samples = _sample(w, dof, loc, n=5000)

    pvals = liu_sf(samples, w, dof, loc, kurtosis=False)[0]
    assert_allclose(median(pvals), 0.5, rtol=1e-2)

    pvals = liu_sf(samples, w, dof, loc, kurtosis=True)[0]
    assert_allclose(median(pvals), 0.5, rtol=1e-2)

    values = [davies_pvalue(s, diag(w)) for s in samples]
    assert_allclose(median(values), 0.5, rtol=1e-2)


def test_pval_calibration2():
    dof = [1, 1, 1]
    w = [128372.0, 23720.11, 528372.0]
    loc = [0.0, 0.0, 0.0]

    random.seed(2)
    samples = _sample(w, dof, loc, n=5000)

    values = [davies_pvalue(s, diag(w)) for s in samples]
    assert_allclose(median(values), 0.5, rtol=1e-2)


def _sample(w, dof, loc, n=1):
    """
    Sample from 𝑋.

    Where
        𝑋 = w[0]⋅χ²(dof[0], loc[1]) + w[1]⋅χ²(dof[1], loc[1]) + ...
    """
    samples = []
    for _ in range(n):
        x = 0.0
        for i, (d, l) in enumerate(zip(dof, loc)):
            x += w[i] * ncx2(d, l).rvs()
        samples.append(x)
    return asarray(samples)
