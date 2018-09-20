import sys
import numpy as np
from numpy import sqrt, atleast_1d, kron, full, inf
from scipy.stats import chi2
from scipy.integrate import quad
from chi2comb import chi2comb_cdf, ChiSquared


def skat_davies_pvalue(
    pmin_q, MuQ, VarQ, KerQ, lambda_, VarRemain, Df, tau_rho, r_all, pmin=None
):

    args = (pmin_q, MuQ, VarQ, KerQ, lambda_, VarRemain, Df, tau_rho, r_all)
    re = quad(_skat_davies_function, 0, 40, args, limit=1000, epsabs=10 ** -12)

    # Might want to add this back in
    if re[1] > 1e-6:
        re = _skat_liu_pvalue(
            pmin_q, MuQ, VarQ, KerQ, lambda_, VarRemain, Df, tau_rho, r_all, pmin
        )
        return re

    pvalue = 1 - re[0]
    if pmin is not None:
        if pmin * len(r_all) < pvalue:
            pvalue = pmin * len(r_all)
    return pvalue


def _skat_liu_pvalue(
    pmin_q, MuQ, VarQ, KerQ, lambda_, VarRemain, Df, tau, r_all, pmin=None
):

    assert False
    #   re=integrate(_skat_liu_function, lower=0, upper=40, subdivisions=2000, pmin_q=pmin_q, MuQ=MuQ, VarQ=VarQ, KerQ=KerQ, lambda_ = lambda_,  VarRemain= VarRemain,  Df= Df, tau = tau, r_all=r_all,abs_tol = 10**-25)

    pvalue = 1 - re[0]

    if pmin is not None:
        if pmin * len(r_all) < pvalue:
            pvalue = pmin * len(r_all)

    return pvalue


def _skat_davies_function(
    x, pmin_q, MuQ, VarQ, KerQ, lambda_, VarRemain, Df, tau, r_all
):
    # x = atleast_1d(x)
    # x2d = atleast_2d(x)

    # temp1 = kron(tau, x2d.T)
    temp1 = tau * x

    # ok = r_all != 1
    # temp = (pmin_q[ok] - temp1[ok]) / (1 - r_all[ok])
    temp = np.divide(
        pmin_q - temp1, 1 - r_all, out=full(temp1.shape, inf), where=r_all != 1.0
    )
    # temp.min = apply(temp, 2, min)
    temp_min = np.min(temp)

    re = 0
    min1 = temp_min
    if min1 > sum(lambda_) * 10 ** 4:
        temp = 0
    else:
        min1_temp = min1 - MuQ
        sd1 = sqrt(VarQ - VarRemain) / sqrt(VarQ)
        min1_st = min1_temp * sd1 + MuQ
        # dav_re = davies(min1_st, lambda_, acc=10 ** (-6))

        chi2s = [ChiSquared(w, 0., 1) for w in lambda_]
        dav_re = chi2comb_cdf(min1_st, chi2s, 0., lim=10000, atol=10 ** -6)

        temp = 1 - dav_re[0]
        if dav_re[1] != 0:
            sys.exit("dav_re$ifault is not 0")
    if temp > 1:
        temp = 1
    # re[i] = (1 - temp) * dchisq(x[i], df=1)
    re = (1 - temp) * chi2(df=1).pdf(x)
    return re


def _skat_liu_function(x, pmin_q, MuQ, VarQ, KerQ, lambda_, VarRemain, Df, tau, r_all):
    x = atleast_1d(x)

    temp1 = kron(tau, x.T)

    temp = (pmin_q - temp1) / (1 - r_all)
    temp_min = apply(temp, 2, min)

    temp_q = (temp_min - MuQ) / sqrt(VarQ) * sqrt(2 * Df) + Df
    re = pchisq(temp_q, df=Df) * dchisq(x, df=1)

    return re

