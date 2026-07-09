"""Tests for ppspecial hypergeometric functions."""

import math

import pytest

from ppspecial import hyp0f1


def close(a, b, rtol=1e-12, atol=1e-14):
    return abs(a - b) <= atol + rtol * abs(b)


class TestHyp0f1:
    def test_zero_argument(self):
        for b in [0.5, 1.0, 2.5, 10.0]:
            assert hyp0f1(b, 0.0) == 1.0

    def test_scipy_documented_value(self):
        assert close(hyp0f1(1.0, 1.0), 2.2795853023360673)

    def test_cos_identity(self):
        # 0F1(; 1/2; -z^2/4) = cos(z)
        for z in [0.25, 0.5, 1.0, 2.0, 3.0]:
            x = -0.25 * z * z
            assert close(hyp0f1(0.5, x), math.cos(z), rtol=2e-13)

    def test_sinc_identity(self):
        # 0F1(; 3/2; -z^2/4) = sin(z) / z
        for z in [0.25, 0.5, 1.0, 2.0, 3.0]:
            x = -0.25 * z * z
            assert close(hyp0f1(1.5, x), math.sin(z) / z, rtol=2e-13)

    def test_cosh_identity(self):
        # 0F1(; 1/2; z^2/4) = cosh(z)
        for z in [0.25, 0.5, 1.0, 2.0, 3.0]:
            x = 0.25 * z * z
            assert close(hyp0f1(0.5, x), math.cosh(z), rtol=2e-13)

    def test_sinhc_identity(self):
        # 0F1(; 3/2; z^2/4) = sinh(z) / z
        for z in [0.25, 0.5, 1.0, 2.0, 3.0]:
            x = 0.25 * z * z
            assert close(hyp0f1(1.5, x), math.sinh(z) / z, rtol=2e-13)

    def test_nonpositive_integer_bottom_parameter(self):
        for b in [0.0, -0.0, -1.0, -3.0]:
            assert hyp0f1(b, 1.0) == pytest.approx(1.0e308)
