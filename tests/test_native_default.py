"""Default import behavior for optional native acceleration."""

import subprocess
import sys
import textwrap


def test_package_import_prefers_available_native_module():
    script = textwrap.dedent(
        """
        import sys
        import types

        native = types.ModuleType("ppspecial_native")
        native.erf = lambda x: "native-erf"
        native.lgamma = lambda x: "native-lgamma"
        native.expit = lambda x: "native-expit"
        sys.modules["ppspecial_native"] = native

        import ppspecial

        assert ppspecial.__native_available__ is True
        assert ppspecial.__native_module__ is native
        assert ppspecial.erf(1.0) == "native-erf"
        assert ppspecial.gammaln(1.0) == "native-lgamma"
        assert ppspecial.sigmoid(1.0) == "native-expit"
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)
