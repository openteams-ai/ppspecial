"""Plain C ABI shared-library smoke tests (ROADMAP Target 1).

This is intentionally separate from tests/test_native_ext.py.  These tests
load the package shared library with ctypes and call exported C functions
directly, without importing a CPython extension module or NumPy ufunc layer.
"""

from __future__ import annotations

import ctypes
import math
import shutil
from pathlib import Path

import pytest

import ppspecial

cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")

pytestmark = pytest.mark.skipif(cc is None, reason="No C compiler available")


@pytest.fixture(scope="module")
def native_lib(tmp_path_factory):
    from postpython.build import build_file

    out_dir = tmp_path_factory.mktemp("ppspecial-native-abi")
    lib_path = build_file(
        Path(ppspecial.__file__),
        output=out_dir / "ppspecial.so",
    )
    return ctypes.CDLL(str(lib_path))


def _bind(lib, source_name: str, argtypes):
    """Bind a source-level POST function through its emitted C symbol."""
    from postpython.compiler.backend.c_backend import c_symbol

    fn = getattr(lib, c_symbol(source_name))
    fn.argtypes = argtypes
    fn.restype = ctypes.c_double
    return fn


UNARY_CASES = [
    ("erf", [1.0], math.erf(1.0), 1e-8),
    ("erfc", [1.0], math.erfc(1.0), 1e-8),
    ("erfinv", [0.5], ppspecial.erfinv(0.5), 1e-15),
    ("lgamma", [5.0], math.lgamma(5.0), 1e-14),
    ("gamma", [5.0], 24.0, 1e-12),
    ("digamma", [1.0], ppspecial.digamma(1.0), 1e-14),
    ("j0", [0.0], 1.0, 1e-8),
    ("j1", [0.0], 0.0, 1e-15),
    ("i0", [1.0], ppspecial.i0(1.0), 1e-14),
    ("k0", [1.0], ppspecial.k0(1.0), 1e-14),
    ("ndtr", [0.0], 0.5, 1e-15),
    ("ndtri", [0.5], 0.0, 1e-15),
    ("expit", [0.0], 0.5, 1e-15),
    ("logit", [0.25], math.log(0.25 / 0.75), 1e-15),
]


@pytest.mark.parametrize("name,args,expected,tol", UNARY_CASES, ids=[c[0] for c in UNARY_CASES])
def test_package_shared_library_exports_representative_unary_kernels(
    native_lib,
    name,
    args,
    expected,
    tol,
):
    fn = _bind(native_lib, name, [ctypes.c_double])
    got = fn(*args)
    assert got == pytest.approx(expected, abs=tol, rel=tol)


def test_package_shared_library_exports_binary_kernels(native_lib):
    beta = _bind(native_lib, "beta", [ctypes.c_double, ctypes.c_double])
    assert beta(2.0, 3.0) == pytest.approx(1.0 / 12.0, rel=1e-12)

    xlogy = _bind(native_lib, "xlogy", [ctypes.c_double, ctypes.c_double])
    assert xlogy(2.0, 3.0) == pytest.approx(2.0 * math.log(3.0), rel=1e-15)


def test_package_shared_library_exports_mixed_int_float_kernel(native_lib):
    polygamma = _bind(native_lib, "polygamma", [ctypes.c_int64, ctypes.c_double])
    assert polygamma(0, 1.0) == pytest.approx(ppspecial.digamma(1.0), rel=1e-14)


def test_libm_colliding_source_names_use_emitted_c_symbols(native_lib):
    from postpython.compiler.backend.c_backend import c_symbol

    for source_name in ["erf", "erfc", "gamma", "lgamma", "j0", "j1"]:
        emitted = c_symbol(source_name)
        assert emitted != source_name
        getattr(native_lib, emitted)
