"""Plain C ABI shared-library smoke tests (ROADMAP Target 1).

This is intentionally separate from tests/test_native_ext.py.  These tests
load the package shared library with ctypes and call exported C functions
directly, without importing a CPython extension module or NumPy ufunc layer.
"""

from __future__ import annotations

import ctypes
import json
import math
import shutil
from pathlib import Path

import pytest

import ppspecial

cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")

pytestmark = pytest.mark.skipif(cc is None, reason="No C compiler available")


@pytest.fixture(scope="module")
def native_artifact(tmp_path_factory):
    from postpython.build import build_file

    out_dir = tmp_path_factory.mktemp("ppspecial-native-abi")
    lib_path = build_file(
        Path(ppspecial.__file__),
        output=out_dir / "ppspecial.so",
        emit_header=True,
        emit_manifest=True,
    )
    return {
        "path": lib_path,
        "lib": ctypes.CDLL(str(lib_path)),
        "header": lib_path.with_suffix(".h").read_text(),
        "manifest": json.loads(lib_path.with_suffix(".json").read_text()),
    }


def _bind(lib, export_name: str, argtypes):
    """Bind a source-level POST export through its stable pp_* C symbol."""
    fn = getattr(lib, f"pp_{export_name}")
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
    native_artifact,
    name,
    args,
    expected,
    tol,
):
    fn = _bind(native_artifact["lib"], name, [ctypes.c_double])
    got = fn(*args)
    assert got == pytest.approx(expected, abs=tol, rel=tol)


def test_package_shared_library_exports_binary_kernels(native_artifact):
    beta = _bind(native_artifact["lib"], "beta", [ctypes.c_double, ctypes.c_double])
    assert beta(2.0, 3.0) == pytest.approx(1.0 / 12.0, rel=1e-12)

    xlogy = _bind(native_artifact["lib"], "xlogy", [ctypes.c_double, ctypes.c_double])
    assert xlogy(2.0, 3.0) == pytest.approx(2.0 * math.log(3.0), rel=1e-15)


def test_package_shared_library_exports_mixed_int_float_kernel(native_artifact):
    polygamma = _bind(native_artifact["lib"], "polygamma", [ctypes.c_int64, ctypes.c_double])
    assert polygamma(0, 1.0) == pytest.approx(ppspecial.digamma(1.0), rel=1e-14)


def test_libm_colliding_source_names_have_stable_pp_symbols(native_artifact):
    lib = native_artifact["lib"]

    for source_name in ["erf", "erfc", "gamma", "lgamma", "j0", "j1"]:
        getattr(lib, f"pp_{source_name}")


def test_function_aliases_are_real_c_exports(native_artifact):
    lib = native_artifact["lib"]
    pp_lgamma = _bind(lib, "lgamma", [ctypes.c_double])
    pp_gammaln = _bind(lib, "gammaln", [ctypes.c_double])
    assert pp_gammaln(5.0) == pytest.approx(pp_lgamma(5.0), rel=1e-15)

    pp_expit = _bind(lib, "expit", [ctypes.c_double])
    pp_sigmoid = _bind(lib, "sigmoid", [ctypes.c_double])
    assert pp_sigmoid(-2.0) == pytest.approx(pp_expit(-2.0), rel=1e-15)


def test_header_declares_representative_stable_exports(native_artifact):
    header = native_artifact["header"]
    assert "double pp_erf(double x);" in header
    assert "double pp_gamma(double x);" in header
    assert "double pp_gammaln(double x);" in header
    assert "double pp_sigmoid(double x);" in header
    assert "double pp_polygamma(int64_t n, double x);" in header


def test_manifest_describes_exported_abi(native_artifact):
    manifest = native_artifact["manifest"]
    assert manifest["post_abi"] == 1
    assert manifest["artifact"] == "ppspecial"

    exports = {entry["name"]: entry for entry in manifest["exports"]}
    assert exports["gamma"]["c_symbol"] == "pp_gamma"
    assert exports["gamma"]["kernel_symbol"] == "__pp_gamma"
    assert exports["j0"]["c_symbol"] == "pp_j0"
    assert exports["gammaln"]["kind"] == "alias"
    assert exports["gammaln"]["alias_of"] == "lgamma"
    assert exports["sigmoid"]["kind"] == "alias"
    assert exports["sigmoid"]["alias_of"] == "expit"
