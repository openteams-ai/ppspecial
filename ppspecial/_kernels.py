"""Compiler-friendly aggregation entrypoint for ppspecial kernels."""

from ppspecial._erf import (
    erf,
    erfc,
    erfinv,
    erfcinv,
)

from ppspecial._gamma import (
    lgamma,
    gammaln,
    gamma,
    digamma,
    polygamma,
    beta,
    lbeta,
    rgamma,
)

from ppspecial._bessel import (
    j0, j1,
    y0, y1,
    i0, i1,
    k0, k1,
)

from ppspecial._stats import (
    ndtr,
    log_ndtr,
    ndtri,
    expit,
    sigmoid,
    log_expit,
    logit,
    xlogy,
    xlog1py,
)

__all__ = [
    # erf
    "erf", "erfc", "erfinv", "erfcinv",
    # gamma
    "lgamma", "gammaln", "gamma", "digamma", "polygamma", "beta", "lbeta", "rgamma",
    # bessel
    "j0", "j1", "y0", "y1", "i0", "i1", "k0", "k1",
    # stats
    "ndtr", "log_ndtr", "ndtri",
    "expit", "sigmoid", "log_expit", "logit",
    "xlogy", "xlog1py",
]
