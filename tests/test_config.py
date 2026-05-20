import numpy as np

from ics_thermal.config import configure_test_case


def test_config_A_shapes():
    C0, C1, k1, k2, T_low, T_high, amp, period, phase = configure_test_case("A", 2)
    assert C0.shape == (2,)
    assert C1.shape == (2,)
    assert T_low.shape == (2,)
    assert T_high.shape == (2,)
    assert amp.shape == (2,)
    assert period.shape == (2,)
    assert phase.shape == (2,)
    assert k1 > 0.0
    assert k2 > 0.0


def test_config_B_length():
    N = 10
    C0, C1, k1, k2, T_low, T_high, amp, period, phase = configure_test_case("B", N)
    assert C0.shape == (N,)
    assert np.allclose(period, 5.0)
