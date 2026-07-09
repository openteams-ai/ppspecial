"""Hypergeometric functions - ppspecial._hyper

Public API
----------
hyp0f1(b, x) - confluent hypergeometric limit function 0F1(; b; x)

Algorithms
----------
hyp0f1 : direct power series

    0F1(; b; x) = sum(k=0..inf) x**k / ((b)_k k!)

This first slice intentionally covers real f64 arguments over a conservative
domain. It is written as a bounded scalar POST loop so compiler behavior around
convergence and non-convergence remains visible to the roadmap. SciPy/XSF uses
a Bessel/gamma-based piecewise implementation for real arguments; ppspecial
needs arbitrary-order Bessel functions and standardized exceptional-value
semantics before adopting that shape.
"""

from postyp import f64, i64
from postpyc import vectorize
from postpyc.math import fabs, floor


@vectorize
def hyp0f1(b: f64, x: f64) -> f64:
    """Confluent hypergeometric limit function 0F1(; b; x), real f64.

    The defining series is entire in x when b is not a non-positive integer.
    Returns +inf-like sentinel for singular bottom parameters or overflow,
    matching the current ppspecial convention for domain errors.
    """
    if b <= 0.0 and b == floor(b):
        return 1.0e308

    if x == 0.0:
        return 1.0

    total: f64 = 1.0
    term: f64 = 1.0
    k: i64 = 1
    max_iter: i64 = 400
    tol: f64 = 1.0e-16

    while k <= max_iter:
        denom: f64 = (b + (k - 1)) * k
        if denom == 0.0:
            return 1.0e308

        term = term * x / denom
        next_total: f64 = total + term

        scale: f64 = fabs(next_total)
        if scale < 1.0:
            scale = 1.0
        if fabs(term) <= tol * scale:
            return next_total

        total = next_total
        if total > 1.0e300:
            return 1.0e308
        if total < -1.0e300:
            return -1.0e308

        k = k + 1

    return total
