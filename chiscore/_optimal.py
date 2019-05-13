from numpy import asarray, divide, exp, full, inf, log, min, sqrt
from scipy.integrate import quad
from scipy.stats import chi2

from chi2comb import ChiSquared, chi2comb_cdf

_EPSABS = 1e-12


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

    lambda_threshold = sum(w) * 10 ** 4
    chi2s = [ChiSquared(i, 0.0, 1) for i in w]
    args = (q, mu, var, kur, lambda_threshold, remain_var, df, trho, grid, chi2s)
    try:
        u = _find_upper_bound(args)
        re = quad(
            _davies_function, 0, u, args, limit=1000, epsabs=_EPSABS, full_output=1
        )
        pvalue = 1 - re[0]
        if re[1] > 1e-6:
            pvalue = _skat_liu_pvalue(
                q, mu, var, kur, w, remain_var, df, trho, grid, pmin
            )

    except RuntimeError:
        pvalue = _skat_liu_pvalue(q, mu, var, kur, w, remain_var, df, trho, grid, pmin)

    if pmin is not None:
        if pmin * len(grid) < abs(pvalue):
            pvalue = pmin * len(grid)
    return pvalue


def _skat_liu_pvalue(
    pmin_q, MuQ, VarQ, KerQ, lambda_, VarRemain, Df, tau, r_all, pmin=None
):
    args = (pmin_q, MuQ, VarQ, KerQ, lambda_, VarRemain, Df, tau, r_all)
    re = quad(_skat_liu_func, 0, 40, args, limit=2000, epsabs=_EPSABS, full_output=1)

    pvalue = 1 - re[0]

    if pmin is not None:
        if pmin * len(r_all) < pvalue:
            pvalue = pmin * len(r_all)

    return pvalue


def _find_upper_bound(args):
    v = 0
    u = 80.0
    imax = 1000
    while v <= 1e-14 and imax > 0:
        u /= 2
        v = _davies_function(u, *args)
        imax -= 1

    if imax == 0:
        raise RuntimeError("Could not find an upper bound.")
    return u * 2


def _davies_function_vec(
    x, pmin_q, MuQ, VarQ, KerQ, lambda_, VarRemain, Df, tau, r_all
):
    y = []
    for xi in x:
        y.append(
            _davies_function(
                xi, pmin_q, MuQ, VarQ, KerQ, lambda_, VarRemain, Df, tau, r_all
            )
        )

    return asarray(y, float)


def _davies_function(
    x, pmin_q, MuQ, VarQ, KerQ, lambda_thr, VarRemain, Df, tau, r_all, chi2s
):
    temp1 = tau * x

    temp = divide(
        pmin_q - temp1, 1 - r_all, out=full(temp1.shape, inf), where=r_all != 1.0
    )
    temp_min = min(temp)

    re = 0
    min1 = temp_min
    if min1 > lambda_thr:
        temp = 0
    else:
        min1_temp = min1 - MuQ
        sd1 = sqrt(VarQ - VarRemain) / sqrt(VarQ)
        min1_st = min1_temp * sd1 + MuQ

        dav_re = chi2comb_cdf(min1_st, chi2s, 0.0, lim=10000, atol=10 ** -5)

        temp = 1 - dav_re[0]
        if dav_re[1] != 0:
            msg = "Could not estimate the cdf value: {}".format(str(dav_re))
            raise RuntimeError(msg)
    if temp >= 1:
        re = 0
    else:
        re = (1 - temp) * _chi2_df1_pdf(x)

    return re


def _chi2_df1_pdf(x):
    # gammaln(1 / 2.0) + log(2) / 2.0
    a = 0.918938533204672669540968854562
    return exp(-0.5 * log(x) - x / 2.0 - a)


def _skat_liu_func(x, pmin_q, MuQ, VarQ, KerQ, lambda_, VarRemain, Df, tau, r_all):

    temp1 = tau * x

    temp = (pmin_q - temp1) / (1 - r_all)
    temp_min = min(temp)

    temp_q = (temp_min - MuQ) / sqrt(VarQ) * sqrt(2 * Df) + Df

    return chi2(df=Df).cdf(temp_q) * chi2(df=1).pdf(x)
