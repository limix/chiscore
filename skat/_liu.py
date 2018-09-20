from numpy import concatenate, sqrt
from scipy.stats import chi2


def skat_mod_liu(q, lambda_):

    r = len(lambda_)

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

    muX = l + delta

    sigmaX = sqrt(2) * a

    Q_norm = (q - muQ) / sigmaQ * sqrt(2 * l) + l

    # Qq = pchisq(Q_norm, df=l, lower_tail=False)
    Qq = chi2(df=l).sf(Q_norm)

    # return concatenate([Qq, muQ, sigmaQ, l])
    return (Qq, muQ, sigmaQ, l)


def main():
    import numpy as np

    data = np.load("skat_mod_liu.npz")
    print(skat_mod_liu(data["args"][0][0], data["args"][1]))
    print(data["pliumod"])
    # np.savez("skat_mod_liu.npz", args=args, pliumod=pliumod)


if __name__ == "__main__":
    main()
