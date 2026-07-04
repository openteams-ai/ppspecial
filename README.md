# ppspecial

`ppspecial` is a reimplementation of `scipy.special`, written entirely in
[POST Python](https://github.com/openteams-ai/postpython) — Performance
Optimized Statically Typed Python. Every function is an ordinary,
fully-typed `.py` kernel that runs under the standard CPython interpreter
**and** compiles ahead-of-time to native shared-library code with the POST
Python reference compiler.

The project serves two purposes:

1. A practical special-function library with a scipy-compatible API surface.
2. The flagship, real-world consumer of the
   [POST Python standard](https://github.com/openteams-ai/postpython/blob/main/docs/spec.md)
   — living proof that a numerical library can be written once in a typed
   Python subset and both interpreted and compiled.

## What a kernel looks like

```python
from postyp import Float64
from postpython import vectorize
from postpython.math import exp

@vectorize
def expit(x: Float64) -> Float64:
    """Logistic sigmoid: 1 / (1 + e^{-x}), numerically stable."""
    if x >= 0.0:
        z: Float64 = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)
```

Under CPython this is immediately callable (scalars, or NumPy arrays with
broadcasting when NumPy is installed). The POST Python compiler lowers the
same source to a C99 NumPy-ufunc loop in a native shared library.

## Implemented functions

| Family | Module | Functions |
|---|---|---|
| Error functions | `_erf` | `erf`, `erfc`, `erfinv`, `erfcinv` |
| Gamma functions | `_gamma` | `lgamma`/`gammaln`, `gamma`, `digamma`, `polygamma`, `beta`, `lbeta`, `rgamma` |
| Bessel functions | `_bessel` | `j0`, `j1`, `y0`, `y1`, `i0`, `i1`, `k0`, `k1` |
| Statistical | `_stats` | `ndtr`, `log_ndtr`, `ndtri`, `expit`/`sigmoid`, `log_expit`, `logit`, `xlogy`, `xlog1py` |

Accuracy targets are documented per module (typically ≤ 1.7e-8 relative
error for the Bessel family, ≤ 1.2e-7 for `erfc`, full float64 for
`erfinv` after Halley polish). The test suite validates against published
reference values.

## Installation

### With pip

```bash
python -m pip install "ppspecial @ git+https://github.com/openteams-ai/ppspecial.git"
```

For development (adds `pytest` and `numpy`):

```bash
git clone https://github.com/openteams-ai/ppspecial.git
cd ppspecial
python -m pip install -e ".[dev]"
```

### With pixi

The repository is a [pixi](https://pixi.sh/) workspace that also installs a
C compiler for native builds:

```bash
pixi install -e dev
pixi run -e dev test            # run the test suite (interpreted + compiled)
pixi run -e dev build-native    # compile the kernels to a plain C shared library
pixi run -e dev build-ext       # build ppspecial_native, a NumPy-ufunc extension
```

## Usage

```python
from ppspecial import erf, gamma, j0, ndtr

erf(1.0)          # 0.8427007929497149
gamma(5.0)        # 24.0
j0(2.404825557695773)   # ~0 (first zero of J0)

import numpy as np
ndtr(np.linspace(-3, 3, 7))   # broadcasts elementwise
```

## Native compilation status

`pixi run build-native` (or `python scripts/build_native.py`) compiles each
kernel module to a native shared library with the reference compiler:

| Module | Status |
|---|---|
| `_erf` | ✅ compiles natively |
| `_bessel` | ✅ compiles natively |
| `_gamma` | ✅ compiles natively |
| `_stats` | ✅ compiles natively (links against `_erf`'s compiled `erfc`/`erfinv` via cross-module POST compilation) |

The entire package also builds as one shared library:
`build_file("ppspecial/__init__.py")` compiles all four translation units
dependencies-first and links them into a single artifact containing all
lowered public kernel definitions. The supported C ABI is the stable
`pp_<name>` namespace described by the generated `ppspecial.h` and
`ppspecial.json` sidecars. Kernel symbols underneath may still be
compiler-mangled to avoid libc/libm collisions, but C-compatible consumers
should call the `pp_*` exports.

## Compiled extension module (NumPy ufuncs)

`pixi run build-ext` (or `python scripts/build_ext.py`) builds
`ppspecial_native` — an importable CPython extension in which every public
function is a real `numpy.ufunc` registered from the compiled kernels:

```python
import numpy as np
import ppspecial_native as pps

type(pps.erf)                      # <class 'numpy.ufunc'>
pps.erf(np.linspace(-3, 3, 7))     # native speed, NumPy broadcasting
pps.ndtr(x, out=buffer)            # out=, where=, dtype dispatch — all ufunc goodies
help(pps.ndtr)                     # shows the original POST Python docstring
```

The extension delegates to the same compiled translation units as the plain
shared library — one set of kernels, three consumption paths (interpreted
Python, C ABI via ctypes/cffi, NumPy ufuncs). The test suite verifies that
compiled ufuncs agree with interpreted mode across sample grids, including
broadcasting.

One code base, one artifact per audience. As the reference compiler grows
(module-level constants next), this library inherits each improvement
without source changes.

> **Note on constants:** polynomial coefficients currently live inside the
> kernel functions rather than at module scope, because the reference
> compiler does not lower module-level constants yet. They will move back
> to module scope when that lands; the spec already permits them.

## Roadmap

The detailed cooperative roadmap is maintained in
[`ROADMAP.md`](ROADMAP.md). It breaks the project into publishable targets
for ppspecial library work, native C ABI work, NumPy extension work, and
POST Python compiler/spec requests discovered from this library.

- Hypergeometric functions: `hyp1f1`, `hyp2f1`, `hyp0f1`
- Orthogonal polynomials: `eval_legendre`, `eval_hermite`, `eval_chebyt`, …
- Elliptic integrals: `ellipk`, `ellipe`, `ellipj`
- Airy functions: `airy`, `airye`
- Incomplete beta/gamma and distribution helpers: `betainc`, `gammainc`, `stdtr`, …
- Compiled-vs-`scipy.special` benchmark suite

## Relationship to POST Python

POST Python is a specification for a compilable, statically-typed subset of
Python, with a reference checker and compiler. `ppspecial` intentionally
contains **no compiler-specific code** — just typed Python that conforms to
the spec's POST Core and POST Ufunc ABI profiles. Any conforming compiler
should be able to build it.

Issues that ppspecial surfaces in the reference compiler get fixed in
[openteams-ai/postpython](https://github.com/openteams-ai/postpython); this
repository stays pure POST Python.
