import sys
import numpy as np
from numpy import concatenate as c
from numpy import (
    where,
    mean,
    zeros,
    sqrt,
    atleast_1d,
    asarray,
    exp,
    kron,
    atleast_2d,
    full,
    inf,
)
from numpy.linalg import eigvalsh
from scipy.stats import chi2
from scipy.integrate import quad
from chi2comb import chi2comb_cdf, ChiSquared


def Get_Davies_PVal(Q, K, Q_resampling=None):

    Q = asarray(atleast_1d(Q), float)

    if Q_resampling is not None:
        Q_all = c(Q, Q_resampling)
    else:
        Q_all = Q

    re = Get_PValue(K, Q_all)
    param = dict()
    param["liu_pval"] = re["p_val_liu"][0]
    param["Is_Converged"] = re["is_converge"][0]

    p_value_resampling = None

    re = dict(
        p_value=re["p_value"][0],
        param=param,
        p_value_resampling=p_value_resampling,
        pval_zero_msg=re["pval_zero_msg"],
    )
    return re


def Get_PValue(K, Q):

    lambda_ = Get_Lambda(K)
    return Get_PValue_Lambda(lambda_, Q)


def Get_PValue_Lambda(lambda_, Q):

    n1 = len(Q)

    p_val = zeros(n1)
    p_val_liu = zeros(n1)
    is_converge = zeros(n1)

    p_val_liu = Get_Liu_PVal_MOD_Lambda(Q, lambda_)

    for i in range(n1):
        chi2s = [ChiSquared(w, 0., 1) for w in lambda_]
        out = chi2comb_cdf(Q[i], chi2s, 0., lim=10000, atol=10 ** -6)

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

        param = Get_Liu_Params_Mod_Lambda(lambda_)
        p_val_msg = Get_Liu_PVal_MOD_Lambda_Zero(
            Q[0],
            param["muQ"],
            param["muX"],
            param["sigmaQ"],
            param["sigmaX"],
            param["ll"],
            param["d"],
        )
        p_val_log = Get_Liu_PVal_MOD_Lambda(Q[0], lambda_, log_p=True)[0]

    return dict(
        p_value=p_val,
        p_val_liu=p_val_liu,
        is_converge=is_converge,
        p_val_log=p_val_log,
        pval_zero_msg=p_val_msg,
    )


def Get_Lambda(K):

    lambda1 = eigvalsh(K)
    lambda1 = np.sort(lambda1)
    IDX1 = where(lambda1 >= 0)[0]

    # eigenvalue bigger than sum(eigenvalues)/1000
    IDX2 = where(lambda1 > mean(lambda1[IDX1]) / 100000)[0]

    if len(IDX2) == 0:
        sys.exit("No Eigenvalue is bigger than 0!!")
    return lambda1[IDX2]


def Get_Liu_PVal_MOD_Lambda(Q_all, lambda_, log_p=False):

    param = Get_Liu_Params_Mod_Lambda(lambda_)

    Q_Norm = (Q_all - param["muQ"]) / param["sigmaQ"]
    Q_Norm1 = Q_Norm * param["sigmaX"] + param["muX"]

    if log_p:
        assert False
        # Q_Norm1 = exp(Q_Norm1)
    p_value = 1 - chi2.cdf(Q_Norm1, df=param["ll"], loc=param["d"])
    # p_value = pchisq(Q_Norm1, df=param.ll, ncp=param.d, lower_tail=False, log_p=log_p)

    return p_value


def Get_Liu_Params_Mod_Lambda(lambda_):
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
        ll = 1. / s2
        a = sqrt(ll)
        d = 0

    muX = ll + d
    sigmaX = sqrt(2) * a

    return dict(ll=ll, d=d, muQ=muQ, muX=muX, sigmaQ=sigmaQ, sigmaX=sigmaX)


def Get_Liu_PVal_MOD_Lambda_Zero(Q, muQ, muX, sigmaQ, sigmaX, l, d):

    Q_Norm = (Q - muQ) / sigmaQ

    temp = c(
        0.05,
        10 ** -10,
        10 ** -20,
        10 ** -30,
        10 ** -40,
        10 ** -50,
        10 ** -60,
        10 ** -70,
        10 ** -80,
        10 ** -90,
        10 ** -100,
    )

    out = qchisq(temp, df=l, ncp=d, lower_tail=False)

    IDX = max(where(out < Q.Norm1))

    pval_msg = "Pvalue < %e" % temp[IDX]
    return pval_msg


def SKAT_Optimal_PValue_Davies(
    pmin_q, MuQ, VarQ, KerQ, lambda_, VarRemain, Df, tau_rho, r_all, pmin=None
):

    args = (pmin_q, MuQ, VarQ, KerQ, lambda_, VarRemain, Df, tau_rho, r_all)
    # SKAT_Optimal_Integrate_Func_Davies(0.08685674, *args)
    re = quad(
        SKAT_Optimal_Integrate_Func_Davies, 0, 40, args, limit=1000, epsabs=10 ** -12
    )
    #   re=try(integrate(SKAT_Optimal_Integrate_Func_Davies, lower=0, upper=40, subdivisions=1000, pmin_q=pmin_q, MuQ=MuQ, VarQ=VarQ, KerQ=KerQ, lambda_ = lambda_,  VarRemain= VarRemain,  Df= Df, tau = tau_rho, r_all=r_all, abs.tol = 10^-25), silent = True)

    # Might want to add this back in
    # if type(re) == "try-error":
    if re[1] > 1e-6:
        re = SKAT_Optimal_PValue_Liu(
            pmin_q, MuQ, VarQ, KerQ, lambda_, VarRemain, Df, tau_rho, r_all, pmin
        )
        return re

    pvalue = 1 - re[0]
    if pmin is not None:
        if pmin * len(r_all) < pvalue:
            pvalue = pmin * len(r_all)
    return pvalue


def SKAT_Optimal_PValue_Liu(
    pmin_q, MuQ, VarQ, KerQ, lambda_, VarRemain, Df, tau, r_all, pmin=None
):

    #   re=integrate(SKAT_Optimal_Integrate_Func_Liu, lower=0, upper=40, subdivisions=2000, pmin_q=pmin_q, MuQ=MuQ, VarQ=VarQ, KerQ=KerQ, lambda_ = lambda_,  VarRemain= VarRemain,  Df= Df, tau = tau, r_all=r_all,abs_tol = 10**-25)

    pvalue = 1 - re[0]

    if pmin is not None:
        if pmin * len(r_all) < pvalue:
            pvalue = pmin * len(r_all)

    return pvalue


def SKAT_Optimal_Integrate_Func_Davies(
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


def SKAT_Optimal_Integrate_Func_Liu(
    x, pmin_q, MuQ, VarQ, KerQ, lambda_, VarRemain, Df, tau, r_all
):
    x = atleast_1d(x)

    temp1 = kron(tau, x.T)

    temp = (pmin_q - temp1) / (1 - r_all)
    temp_min = apply(temp, 2, min)

    temp_q = (temp_min - MuQ) / sqrt(VarQ) * sqrt(2 * Df) + Df
    re = pchisq(temp_q, df=Df) * dchisq(x, df=1)

    return re


def test_SKAT_Optimal_PValue_Davies():
    data = np.load("SKAT_Optimal_PValue_Davies.npz")
    pval = SKAT_Optimal_PValue_Davies(*data["args"])
    assert abs(pval - data["pval"]) < 1e-9


def test_Get_Davies_PVal():
    data = np.load("Get_Davies_PVal.npz")
    pval = Get_Davies_PVal(*data["args"])["p_value"]
    assert abs(pval - data["pval"]) < 1e-9


if __name__ == "__main__":
    test_Get_Davies_PVal()
    test_SKAT_Optimal_PValue_Davies()
