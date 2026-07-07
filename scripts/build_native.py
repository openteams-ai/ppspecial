"""Compile each ppspecial kernel module to a native shared library.

Every module must build (the reference compiler supports cross-module
lowering and linking, so _stats links against _erf's compiled erfc and
erfinv). The whole package is also built into a single shared library
from ppspecial/__init__.py. Any failure exits non-zero.
"""

import sys
import tempfile
from pathlib import Path

from postpyc.build import build_file, BuildError

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "ppspecial"

EXPECTED_NATIVE = ["_erf", "_bessel", "_gamma", "_stats"]


def main() -> int:
    out_dir = Path(tempfile.mkdtemp(prefix="ppspecial-native-"))
    failures = []

    for name in EXPECTED_NATIVE:
        source = PACKAGE_DIR / f"{name}.py"
        try:
            lib = build_file(source, output=out_dir / f"{name}.so")
            print(f"  {name:10s} OK       -> {lib}")
        except BuildError as exc:
            failures.append(name)
            print(f"  {name:10s} FAILED")
            print("    " + "\n    ".join(str(exc).splitlines()[:6]))

    # The full package: one shared library with all four translation units
    # plus stable C ABI sidecars (`ppspecial.h`, `ppspecial.json`).
    try:
        lib = build_file(
            PACKAGE_DIR / "__init__.py",
            output=out_dir / "ppspecial.so",
            emit_header=True,
            emit_manifest=True,
        )
        print(f"  {'package':10s} OK       -> {lib}")
        print(f"  {'header':10s} OK       -> {lib.with_suffix('.h')}")
        print(f"  {'manifest':10s} OK       -> {lib.with_suffix('.json')}")
    except BuildError as exc:
        failures.append("package")
        print(f"  {'package':10s} FAILED")
        print("    " + "\n    ".join(str(exc).splitlines()[:6]))

    if failures:
        print(f"\n{len(failures)} build(s) failed: {failures}")
        return 1
    print("\nAll modules and the full package compile natively.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
