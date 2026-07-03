"""Compiled NumPy extension vs. interpreted mode (ROADMAP Target 2).

Builds ppspecial as a CPython extension module once per session, then
verifies that every registered ufunc agrees with the interpreted POST
Python implementation, including array broadcasting behavior.
"""

import importlib.util
import shutil

import pytest

np = pytest.importorskip("numpy")

import ppspecial

cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")

pytestmark = pytest.mark.skipif(cc is None, reason="No C compiler available")


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    from postpython.build import build_file
    from pathlib import Path

    out_dir = tmp_path_factory.mktemp("ppspecial-ext")
    ext = build_file(
        Path(ppspecial.__file__),
        ext_module=True,
        module_name="ppspecial_native_test",
        output=out_dir / "ppspecial_native_test.so",
    )
    spec = importlib.util.spec_from_file_location("ppspecial_native_test", str(ext))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_public_ufuncs_registered(native):
    registered = {n for n in dir(native) if not n.startswith("_")}
    expected = {
        "erf", "erfc", "erfinv", "erfcinv",
        "lgamma", "gamma", "digamma", "polygamma", "beta", "lbeta",
        "j0", "j1", "y0", "y1", "i0", "i1", "k0", "k1",
        "ndtr", "log_ndtr", "ndtri", "expit", "log_expit", "logit",
        "xlogy", "xlog1py",
    }
    assert expected <= registered
    for name in expected:
        assert isinstance(getattr(native, name), np.ufunc), name


UNARY_CASES = [
    ("erf", np.linspace(-4.0, 4.0, 17)),
    ("erfc", np.linspace(-4.0, 4.0, 17)),
    ("erfinv", np.linspace(-0.95, 0.95, 13)),
    ("lgamma", np.linspace(0.2, 12.0, 13)),
    ("gamma", np.linspace(0.2, 6.0, 13)),
    ("digamma", np.linspace(0.3, 10.0, 11)),
    ("j0", np.linspace(-10.0, 10.0, 21)),
    ("j1", np.linspace(-10.0, 10.0, 21)),
    ("y0", np.linspace(0.2, 12.0, 13)),
    ("i0", np.linspace(-4.0, 4.0, 13)),
    ("k0", np.linspace(0.2, 6.0, 11)),
    ("ndtr", np.linspace(-5.0, 5.0, 21)),
    ("log_ndtr", np.linspace(-30.0, 6.0, 19)),
    ("ndtri", np.linspace(0.02, 0.98, 13)),
    ("expit", np.linspace(-30.0, 30.0, 21)),
    ("logit", np.linspace(0.05, 0.95, 13)),
]


@pytest.mark.parametrize("name,grid", UNARY_CASES, ids=[c[0] for c in UNARY_CASES])
def test_compiled_matches_interpreted(native, name, grid):
    compiled = getattr(native, name)
    interpreted = getattr(ppspecial, name)
    expected = np.array([interpreted(float(v)) for v in grid])
    np.testing.assert_allclose(compiled(grid), expected, rtol=1e-13, atol=1e-300)


def test_binary_ufuncs_match_interpreted(native):
    a = np.linspace(0.5, 4.0, 9)
    b = np.linspace(0.5, 3.0, 9)
    expected = np.array([ppspecial.beta(float(x), float(y)) for x, y in zip(a, b)])
    np.testing.assert_allclose(native.beta(a, b), expected, rtol=1e-12)

    x = np.linspace(0.0, 3.0, 7)
    y = np.linspace(0.5, 2.0, 7)
    expected = np.array([ppspecial.xlogy(float(u), float(v)) for u, v in zip(x, y)])
    np.testing.assert_allclose(native.xlogy(x, y), expected, rtol=1e-14)


def test_broadcasting_matches_numpy_semantics(native):
    # (3, 1) broadcast against (4,) → (3, 4), matching NumPy's rules —
    # something the interpreted wrapper also supports via NumPy.
    col = np.linspace(-1.0, 1.0, 3).reshape(3, 1)
    row = np.linspace(0.5, 2.0, 4)
    out = native.xlogy(col, row)
    assert out.shape == (3, 4)
    expected = ppspecial.xlogy(col, row)
    np.testing.assert_allclose(out, expected, rtol=1e-14)


def test_ufunc_out_parameter_and_docstring(native):
    x = np.linspace(-2.0, 2.0, 9)
    buffer = np.empty_like(x)
    result = native.erf(x, out=buffer)
    assert result is buffer
    assert "error function" in native.erf.__doc__.lower()


def test_mixed_dtype_polygamma(native):
    n = np.zeros(5, dtype=np.int64)
    x = np.linspace(0.5, 4.5, 5)
    expected = np.array([ppspecial.digamma(float(v)) for v in x])
    np.testing.assert_allclose(native.polygamma(n, x), expected, rtol=1e-12)
