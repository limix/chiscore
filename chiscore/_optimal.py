from numpy import sqrt, full, inf, asarray, divide, min
from scipy.stats import chi2
from scipy.integrate import quad
from chi2comb import chi2comb_cdf, ChiSquared


def optimal_davies_pvalue(q, mu, var, kur, w, remain_var, df, trho, grid, pmin=None):
    r"""Joint significance of statistics derived from chi2-squared distributions.

    Parameters
    ----------
    q : array_like
        Most significant of the independent test statistics, qₘᵢₙ(𝜌ᵥ) [1].
    mu : float
        Mean of the linear combination of the chi-squared distributions.
    var : float
        Variance of the linear combination of the chi-squared distributions.
    kur : float
        Kurtosis of the linear combination of the chi-squared distributions.
    w : array_like
        Weights of the linear combination.
    remain_var : float
        Remaining variance assigned to a Normal distribution.
    df : float
        Overall degrees of freedom.
    trho : array_like
        Weight between the combination of chi-squared distributions and independent
        chi-squared distributions.
    grid : array_like
        Grid parameters.
    pmin : float
        Boundary of the possible final p-values.

    Returns
    -------
    float
        Estimated p-value.

    References
    ----------
    [1] Lee, Seunggeun, Michael C. Wu, and Xihong Lin. "Optimal tests for rare variant
    effects in sequencing association studies." Biostatistics 13.4 (2012): 762-775.
    """
    q = asarray(q, float)
    mu = float(mu)
    var = float(var)
    kur = float(kur)

    w = asarray(w, float)
    remain_var = float(remain_var)
    df = float(df)
    trho = asarray(trho, float)
    grid = asarray(grid, float)

    args = (q, mu, var, kur, w, remain_var, df, trho, grid)
    re = quad(_davies_function, 0, 40, args, limit=1000, epsabs=10 ** -12)

    # Might want to add this back in
    if re[1] > 1e-6:
        re = _skat_liu_pvalue(q, mu, var, kur, w, remain_var, df, trho, grid, pmin)
        return re

    pvalue = 1 - re[0]
    if pmin is not None:
        if pmin * len(grid) < pvalue:
            pvalue = pmin * len(grid)
    return pvalue


def _skat_liu_pvalue(
    pmin_q, MuQ, VarQ, KerQ, lambda_, VarRemain, Df, tau, r_all, pmin=None
):
    # TODO: Unit tests are not reaching this function yet.
    assert False
    #   re=integrate(_skat_liu_function, lower=0, upper=40, subdivisions=2000,
    # pmin_q=pmin_q, MuQ=MuQ, VarQ=VarQ, KerQ=KerQ, lambda_ = lambda_,
    # VarRemain= VarRemain,  Df= Df, tau = tau, r_all=r_all,abs_tol = 10**-25)

    # pvalue = 1 - re[0]

    # if pmin is not None:
    #     if pmin * len(r_all) < pvalue:
    #         pvalue = pmin * len(r_all)

    # return pvalue


def _davies_function(x, pmin_q, MuQ, VarQ, KerQ, lambda_, VarRemain, Df, tau, r_all):
    temp1 = tau * x

    temp = divide(
        pmin_q - temp1, 1 - r_all, out=full(temp1.shape, inf), where=r_all != 1.0
    )
    temp_min = min(temp)

    re = 0
    min1 = temp_min
    if min1 > sum(lambda_) * 10 ** 4:
        temp = 0
    else:
        min1_temp = min1 - MuQ
        sd1 = sqrt(VarQ - VarRemain) / sqrt(VarQ)
        min1_st = min1_temp * sd1 + MuQ

        chi2s = [ChiSquared(w, 0., 1) for w in lambda_]
        dav_re = chi2comb_cdf(min1_st, chi2s, 0., lim=10000, atol=10 ** -5)

        temp = 1 - dav_re[0]
        if dav_re[1] != 0:
            msg = "Could not estimate the cdf value: {}".format(str(dav_re))
            raise RuntimeError(msg)
    if temp > 1:
        temp = 1
    re = (1 - temp) * chi2(df=1).pdf(x)
    return re


def _skat_liu_function(x, pmin_q, MuQ, VarQ, KerQ, lambda_, VarRemain, Df, tau, r_all):
    # TODO: Unit tests are not reaching this function yet.
    assert False
    # x = atleast_1d(x)

    # temp1 = kron(tau, x.T)

    # temp = (pmin_q - temp1) / (1 - r_all)
    # temp_min = apply(temp, 2, min)

    # temp_q = (temp_min - MuQ) / sqrt(VarQ) * sqrt(2 * Df) + Df
    # re = pchisq(temp_q, df=Df) * dchisq(x, df=1)

    # return re
