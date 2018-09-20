from numpy import sqrt
from scipy.stats import chi2


def mod_liu(q, w):
    r"""Joint significance of statistics derived from chi2-squared distributions.

    Parameters
    ----------
    q : float
        Test statistics.
    w : array_like
        Weights of the linear combination.

    Returns
    -------
    float
        Estimated p-value.
    """

    c1 = sum(w)

    c2 = sum(w ** 2)

    c3 = sum(w ** 3)

    c4 = sum(w ** 4)

    s1 = c3 / (c2 ** (3 / 2))

    s2 = c4 / c2 ** 2

    muQ = c1

    sigmaQ = sqrt(2 * c2)

    if s1 ** 2 > s2:

        a = 1 / (s1 - sqrt(s1 ** 2 - s2))

        delta = s1 * a ** 3 - a ** 2

        l = a ** 2 - 2 * delta

    else:

        delta = 0
        l = 1 / s2
        a = sqrt(l)

    Q_norm = (q - muQ) / sigmaQ * sqrt(2 * l) + l

    Qq = chi2(df=l).sf(Q_norm)

    return (Qq, muQ, sigmaQ, l)
