"""Compile each ppspecial kernel module to a native shared library.

Reports per-module status. Modules listed in EXPECTED_NATIVE must build;
a failure there exits non-zero. _stats is known to require cross-module
POST compilation (it imports erfc/erfinv from ppspecial._erf), which the
reference compiler does not support yet.
"""

import sys
import tempfile
from pathlib import Path

from postpython.build import build_file, BuildError

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "ppspecial"

EXPECTED_NATIVE = ["_erf", "_bessel", "_gamma"]
KNOWN_BLOCKED = {
    "_stats": "requires cross-module POST compilation (imports from ppspecial._erf)",
}


def main() -> int:
    out_dir = Path(tempfile.mkdtemp(prefix="ppspecial-native-"))
    failures = []

    for name in EXPECTED_NATIVE + list(KNOWN_BLOCKED):
        source = PACKAGE_DIR / f"{name}.py"
        try:
            lib = build_file(source, output=out_dir / f"{name}.so")
            print(f"  {name:10s} OK       -> {lib}")
            if name in KNOWN_BLOCKED:
                print(f"  {name:10s} NOTE: expected to fail but built; update this script.")
        except BuildError as exc:
            if name in KNOWN_BLOCKED:
                print(f"  {name:10s} BLOCKED  ({KNOWN_BLOCKED[name]})")
            else:
                failures.append(name)
                print(f"  {name:10s} FAILED")
                print("    " + "\n    ".join(str(exc).splitlines()[:6]))

    if failures:
        print(f"\n{len(failures)} module(s) failed that should compile: {failures}")
        return 1
    print("\nAll expected modules compile natively.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
