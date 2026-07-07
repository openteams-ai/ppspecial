"""ppspecial — postpyc reimplementation of scipy.special.

Each function is a @vectorize kernel, written in fully-typed postpyc.
The postpyc compiler lowers them to native shared-library code; in
interpreted mode they run via the pure-Python broadcast loop. When the
optional `ppspecial_native` extension module is installed next to this package,
matching public functions are replaced with native NumPy ufuncs at import time.

Function families implemented
------------------------------
Error functions  (_erf)    : erf, erfc, erfinv, erfcinv
Gamma functions  (_gamma)  : lgamma / gammaln, gamma, digamma, polygamma,
                              beta, lbeta, rgamma
Bessel functions (_bessel) : j0, j1, y0, y1, i0, i1, k0, k1
Statistical      (_stats)  : ndtr, log_ndtr, ndtri, expit / sigmoid,
                              log_expit, logit, xlogy, xlog1py

Roadmap (not yet implemented)
------------------------------
Hypergeometric   : hyp1f1, hyp2f1, hyp0f1
Orthogonal poly  : eval_legendre, eval_hermite, eval_chebyshev*, eval_laguerre
Spherical harm.  : sph_harm
Elliptic integrals: ellipk, ellipe, ellipj
Airy functions   : airy, airye
Statistical dists: betainc, betaincinv, stdtr, stdtrit, chdtr, chdtri, ...
"""

from importlib import import_module as _import_module

from ppspecial._erf import (
    erf,
    erfc,
    erfinv,
    erfcinv,
)

from ppspecial._gamma import (
    lgamma,
    gammaln,   # alias for lgamma
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
    sigmoid,   # alias for expit
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

__native_available__ = False
__native_module__ = None


def _prefer_native() -> None:
    """Prefer compiled ufuncs when a sibling native extension is installed."""
    global __native_available__, __native_module__, gammaln, sigmoid

    try:
        native = _import_module("ppspecial_native")
    except ModuleNotFoundError as exc:
        if exc.name == "ppspecial_native":
            return
        raise

    replaced = []
    for name in __all__:
        if hasattr(native, name):
            globals()[name] = getattr(native, name)
            replaced.append(name)

    # The native extension may expose canonical kernels without Python aliases.
    if "gammaln" not in replaced and "lgamma" in replaced:
        gammaln = lgamma
        replaced.append("gammaln")
    if "sigmoid" not in replaced and "expit" in replaced:
        sigmoid = expit
        replaced.append("sigmoid")

    if replaced:
        __native_available__ = True
        __native_module__ = native


_prefer_native()

del _prefer_native, _import_module
