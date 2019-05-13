import sys

import numpy as np
from numpy import asarray, atleast_1d, mean, sqrt, where, zeros
from numpy.linalg import eigvalsh
from scipy.stats import chi2

from chi2comb import ChiSquared, chi2comb_cdf


def davies_pvalue(q, w):
    """
    Joint significance of statistics derived from chi2-squared distributions.

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

    q = asarray(atleast_1d(q), float)
    w = asarray(w, float)

    re = _pvalue_lambda(_lambda(w), q)
    param = dict()
    param["liu_pval"] = re["p_val_liu"][0]
    param["Is_Converged"] = re["is_converge"][0]

    return re["p_value"][0]


def _pvalue_lambda(lambda_, Q):

    n1 = len(Q)

    p_val = zeros(n1)
    p_val_liu = zeros(n1)
    is_converge = zeros(n1)

    p_val_liu = _liu_pvalue_mod_lambda(Q, lambda_)

    for i in range(n1):
        chi2s = [ChiSquared(w, 0.0, 1) for w in lambda_]
        out = chi2comb_cdf(Q[i], chi2s, 0.0, lim=10000, atol=10 ** -6)

        p_val[i] = 1 - out[0]

        is_converge[i] = 1

        # check convergence
        if len(lambda_) == 1:
            p_val[i] = p_val_liu[i]
        elif out[1] != 0:
            is_converge[i] = 0

        # check p-value
        if p_val[i] > 1 or p_val[i] <= 0:
            is_converge[i] = 0
            p_val[i] = p_val_liu[i]

    p_val_msg = None
    p_val_log = None
    if p_val[0] == 0:
        # TODO: the tests have never been here. Come up with a tests that arrives here.
        param = _liu_params_mod_lambda(lambda_)
        p_val_msg = _liu_pvalue_mod_lambda_zero(
            Q[0],
            param["muQ"],
            param["muX"],
            param["sigmaQ"],
            param["sigmaX"],
            param["ll"],
            param["d"],
        )
        p_val_log = _liu_pvalue_mod_lambda(Q[0], lambda_, log_p=True)[0]

    return dict(
        p_value=p_val,
        p_val_liu=p_val_liu,
        is_converge=is_converge,
        p_val_log=p_val_log,
        pval_zero_msg=p_val_msg,
    )


def _lambda(K):
    lambda1 = eigvalsh(K)
    lambda1 = np.sort(lambda1)
    idx1 = where(lambda1 >= 0)[0]

    # eigenvalue bigger than sum(eigenvalues)/1000
    idx2 = where(lambda1 > mean(lambda1[idx1]) / 100000)[0]

    if len(idx2) == 0:
        sys.exit("No Eigenvalue is bigger than 0!!")
    return lambda1[idx2]


def _liu_pvalue_mod_lambda(Q_all, lambda_, log_p=False):

    param = _liu_params_mod_lambda(lambda_)

    Q_Norm = (Q_all - param["muQ"]) / param["sigmaQ"]
    Q_Norm1 = Q_Norm * param["sigmaX"] + param["muX"]

    if log_p:
        assert False
        # Q_Norm1 = exp(Q_Norm1)
    p_value = 1 - chi2.cdf(Q_Norm1, df=param["ll"], loc=param["d"])

    return p_value


def _liu_params_mod_lambda(lambda_):
    # Helper function for getting the parameters for the null approximation

    c1 = zeros(4)
    for i in range(4):
        c1[i] = sum(lambda_ ** (i + 1))

    muQ = c1[0]
    sigmaQ = sqrt(2 * c1[1])
    s1 = c1[2] / c1[1] ** (3 / 2)
    s2 = c1[3] / c1[1] ** 2

    if s1 ** 2 > s2:
        a = 1 / (s1 - sqrt(s1 ** 2 - s2))
        d = s1 * a ** 3 - a ** 2
        ll = a ** 2 - 2 * d
    else:
        ll = 1.0 / s2
        a = sqrt(ll)
        d = 0

    muX = ll + d
    sigmaX = sqrt(2) * a

    return dict(ll=ll, d=d, muQ=muQ, muX=muX, sigmaQ=sigmaQ, sigmaX=sigmaX)


def _liu_pvalue_mod_lambda_zero(Q, muQ, muX, sigmaQ, sigmaX, l, d):

    assert False
    # TODO: Unit tests are not currently testing this function.
    # temp = c(
    #     0.05,
    #     10 ** -10,
    #     10 ** -20,
    #     10 ** -30,
    #     10 ** -40,
    #     10 ** -50,
    #     10 ** -60,
    #     10 ** -70,
    #     10 ** -80,
    #     10 ** -90,
    #     10 ** -100,
    # )

    # out = qchisq(temp, df=l, ncp=d, lower_tail=False)

    # IDX = max(where(out < Q.Norm1))

    # pval_msg = "Pvalue < %e" % temp[IDX]
    # return pval_msg
