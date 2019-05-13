from numpy import asarray, maximum, sqrt, sum
from scipy.stats import ncx2


def liu_sf(t, lambs, dofs, deltas):
    """
    Liu approximation to linear combination of noncentral chi-squared variables.

    Let

        𝑋 = ∑λᵢχ²(hᵢ, 𝛿ᵢ)

    be a linear combination of noncentral chi-squared random variables, where λᵢ, hᵢ,
    and 𝛿ᵢ are the weights, degrees of freedom, and noncentrality parameters.
    [1] proposes a method that approximates 𝑋 by a single noncentral chi-squared
    random variable, χ²(l, 𝛿):

        Pr(𝑋 > 𝑡) ≈ Pr(χ²(𝑙, 𝛿) > 𝑡⁺𝜎ₓ + 𝜇ₓ),

    where 𝑡⁺ = (𝑡 - 𝜇₀) / 𝜎₀, 𝜇ₓ = 𝑙 + 𝛿, and 𝜎ₓ = √(𝑙 + 2𝛿).
    The mean and standard deviation of 𝑋 are given by 𝜇₀ and 𝜎₀.

    Returns
    -------
    q : float, ndarray
        Approximated survival function applied to 𝑡: Pr(𝑋 > 𝑡).
    dof : float
        Degrees of freedom of χ²(𝑙, 𝛿), 𝑙.
    ncen : float
        Noncentrality parameter of χ²(𝑙, 𝛿), 𝛿.
    info : dict
        Additional information: mu_q, sigma_q, mu_x, sigma_x, and t_star.

    Example
    -------
    Let us approximate

        𝑋 = 0.5⋅χ²(1, 1) + 0.4⋅χ²(2, 0.6) + 0.1⋅χ²(1, 0.8),

    and evaluate Pr(𝑋 > 2).

    .. doctest::

        >>> from chiscore import liu_sf
        >>>
        >>> w = [0.5, 0.4, 0.1]
        >>> dofs = [1, 2, 1]
        >>> deltas = [1, 0.6, 0.8]
        >>> (q, dof, delta, _) = liu_sf(2, w, dofs, deltas)
        >>> q  # doctest: +FLOAT_CMP
        0.4577529852208846
        >>> dof  # doctest: +FLOAT_CMP
        3.5556138890755395
        >>> delta  # doctest: +FLOAT_CMP
        0.7491921870025307

    Therefore, we have

        Pr(𝑋 > 2) ≈ Pr(χ²(3.56, 0.75) > 𝑡⁺𝜎ₓ + 𝜇ₓ) = 0.458.

    References
    ----------
    [1] Liu, H., Tang, Y., & Zhang, H. H. (2009). A new chi-square approximation to the
        distribution of non-negative definite quadratic forms in non-central normal
        variables. Computational Statistics & Data Analysis, 53(4), 853-856.
    """
    t = asarray(t, float)
    lambs = asarray(lambs, float)
    dofs = asarray(dofs, float)
    deltas = asarray(deltas, float)

    lambs = {i: lambs ** i for i in range(1, 5)}

    c = {i: sum(lambs[i] * dofs) + i * sum(lambs[i] * deltas) for i in range(1, 5)}

    s1 = c[3] / sqrt(c[2]) ** 3
    s2 = c[4] / c[2] ** 2

    s12 = s1 ** 2
    if s12 > s2:
        a = 1 / (s1 - sqrt(s12 - s2))
        delta_x = s1 * a ** 3 - a ** 2
        dof_x = a ** 2 - 2 * delta_x
    else:
        a = 1 / s1
        delta_x = 0
        dof_x = 1 / s12

    mu_q = c[1]
    sigma_q = sqrt(2 * c[2])

    mu_x = dof_x + delta_x
    sigma_x = sqrt(2 * (dof_x + 2 * delta_x))

    t_star = (t - mu_q) / sigma_q
    tfinal = t_star * sigma_x + mu_x

    q = ncx2.sf(tfinal, dof_x, maximum(delta_x, 1e-9))

    info = {
        "mu_q": mu_q,
        "sigma_q": sigma_q,
        "mu_x": mu_x,
        "sigma_x": sigma_x,
        "t_star": t_star,
    }
    return (q, dof_x, delta_x, info)
