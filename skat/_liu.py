from numpy import sqrt
from scipy.stats import chi2


def mod_liu(q, lambda_):
    # q: test statistic
    # lambda_:eigenvalues (weights of the linear combination...)

    c1 = sum(lambda_)

    c2 = sum(lambda_ ** 2)

    c3 = sum(lambda_ ** 3)

    c4 = sum(lambda_ ** 4)

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
