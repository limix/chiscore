from numpy import sqrt, asarray, sum, maximum
from scipy.stats import ncx2


def liu_sf(t, lambs, dofs, deltas):
    """
    Liu approximation to linear combination of noncentral chi-squared variables.
    """
    t = asarray(t, float)
    lambs = asarray(lambs, float)
    dofs = asarray(dofs, float)
    deltas = asarray(deltas, float)

    lambs = {i: lambs ** i for i in range(1, 5)}

    c = {i: sum(lambs[i] * dofs) + i * sum(lambs[i] * deltas) for i in range(1, 5)}

    s1 = c[3] / sqrt(c[2]) ** 3
    s2 = c[4] / c[2] ** 2

    if s1 ** 2 > s2:
        a = 1 / (s1 - sqrt(s1 ** 2 - s2))
        delta_x = s1 * a ** 3 - a ** 2
        dof_x = a ** 2 - 2 * delta_x
    else:
        a = 1 / s1
        delta_x = 0
        dof_x = 1 / s1 ** 2

    mu_q = c[1]
    sigma_q = sqrt(2 * c[2])

    mu_x = dof_x + delta_x
    sigma_x = sqrt(2 * (dof_x + 2 * delta_x))

    tstar = (t - mu_q) / sigma_q
    tfinal = tstar * sigma_x + mu_x

    p = ncx2.sf(tfinal, dof_x, maximum(delta_x, 1e-9))
    return (p, dof_x, delta_x, mu_q, sigma_q)
