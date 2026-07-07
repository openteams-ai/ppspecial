"""Build ppspecial as an importable CPython extension module.

Produces `ppspecial_native.<ext-suffix>` in the repository root: a real
extension module whose attributes are numpy.ufunc objects registered from
the compiled kernels (postpyc `ext_module=True` output). This target
is intentionally separate from the plain C shared library built by
scripts/build_native.py; both link the same compiled translation units.
"""

import importlib.util
import sys
from pathlib import Path

from postpyc.build import build_file, BuildError

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_ENTRY = REPO_ROOT / "ppspecial" / "_kernels.py"
MODULE_NAME = "ppspecial_native"


def main() -> int:
    try:
        ext_path = build_file(
            KERNEL_ENTRY,
            ext_module=True,
            module_name=MODULE_NAME,
        )
    except BuildError as exc:
        print("extension build FAILED:")
        print("  " + "\n  ".join(str(exc).splitlines()[:8]))
        return 1

    # Move the artifact to the repo root and smoke-test the import.
    target = REPO_ROOT / ext_path.name
    if ext_path != target:
        ext_path.replace(target)

    spec = importlib.util.spec_from_file_location(MODULE_NAME, str(target))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ufuncs = sorted(n for n in dir(module) if not n.startswith("_"))
    print(f"built {target.name}")
    print(f"registered {len(ufuncs)} ufuncs: {', '.join(ufuncs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
