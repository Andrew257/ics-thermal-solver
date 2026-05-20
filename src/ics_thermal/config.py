import numpy as np

from .ics_driver import run_ics_chain
from .pcs_driver import run_pcs_chain


def configure_test_case(test_name: str, N: int):
    """
    Return (C0, C1, k1, k2, T_low, T_high, amp, period, phase)
    for test cases A–F.
    """

    if test_name == "A":
        assert N == 2, "Test A requires N=2"
        C0 = np.array([1.0, 1.0])
        C1 = np.array([5.5, 5.5])
        k1 = 0.001
        k2 = 0.001
        T_low = np.array([47.5, 47.5])
        T_high = np.array([52.5, 52.5])
        amp = np.array([2e3, 2e3])
        period = np.array([5.0, 5.0])
        phase = np.array([0.0, np.pi / 2.0])
        return C0, C1, k1, k2, T_low, T_high, amp, period, phase

    if test_name == "B":
        C0 = np.full(N, 1.0)
        C1 = np.full(N, 5.5)
        k1 = 0.001
        k2 = 0.001
        T_low = np.full(N, 47.5)
        T_high = np.full(N, 52.5)
        amp = np.full(N, 2e3)
        period = np.full(N, 5.0)
        phase = np.linspace(0, 2 * np.pi, N)
        return C0, C1, k1, k2, T_low, T_high, amp, period, phase

    if test_name == "C":
        C0 = np.full(N, 1.0)
        C1 = np.full(N, 5.5)
        k1 = 50.0
        k2 = 50.0
        T_low = np.full(N, 49.9)
        T_high = np.full(N, 50.1)
        amp = np.full(N, 2e6)
        period = np.full(N, 5.0)
        phase = np.linspace(0, 2 * np.pi, N)
        return C0, C1, k1, k2, T_low, T_high, amp, period, phase

    if test_name == "D":
        rng = np.random.default_rng(seed=12345)
        C0 = rng.uniform(0.5, 1.5, N)
        C1 = rng.uniform(2.25, 8.25, N)
        k1 = rng.uniform(0.0005, 0.0015)
        k2 = rng.uniform(0.0005, 0.0015)
        T_low = rng.uniform(46.0, 49.0, N)
        T_high = rng.uniform(51.0, 53.0, N)
        amp = rng.uniform(1e3, 3e3, N)
        period = rng.uniform(2.5, 7.5, N)
        phase = rng.uniform(0, 2 * np.pi, N)
        return C0, C1, k1, k2, T_low, T_high, amp, period, phase

    if test_name == "E":
        C0 = np.full(N, 1.0)
        C1 = np.full(N, 5.5)
        k1 = 0.001
        k2 = 0.001
        T_low = np.full(N, 47.5)
        T_high = np.full(N, 52.5)
        amp = np.zeros(N)
        amp[0::2] = +2e3
        amp[1::2] = -2e3
        period = np.full(N, 5.0)
        phase = np.zeros(N)
        phase[1::2] = np.pi
        return C0, C1, k1, k2, T_low, T_high, amp, period, phase

    if test_name == "F":
        C0 = np.full(N, 1.0)
        C1 = np.full(N, 5.5)
        k1 = 0.001
        k2 = 0.001

        T_low = np.full(N, -np.inf)
        T_high = np.full(N, +np.inf)
        T_low[0] = 47.5
        T_high[0] = 52.5
        T_low[N - 1] = 47.5
        T_high[N - 1] = 52.5

        amp = np.full(N, 2e3)
        period = np.full(N, 5.0)
        phase = np.linspace(0, 4 * np.pi, N)
        return C0, C1, k1, k2, T_low, T_high, amp, period, phase

    raise ValueError(f"Unknown test case: {test_name}")


def run_test_case(
    test_name: str,
    N: int,
    solver: str = "both",
    t_final: float = 13.0,
    dt_ics: float = 0.05,
    h_pcs: float = 5e-6,
    log: bool = False,
):
    params = configure_test_case(test_name, N)

    if solver.upper() == "ICS":
        ics = run_ics_chain(N, params, t_final=t_final, dt_ics=dt_ics, log=log)
        return ics

    if solver.upper() == "PCS":
        pcs = run_pcs_chain(N, params, t_final=t_final, h_pcs=h_pcs)
        return pcs

    ics = run_ics_chain(N, params, t_final=t_final, dt_ics=dt_ics, log=log)
    pcs = run_pcs_chain(N, params, t_final=t_final, h_pcs=h_pcs)
    return ics, pcs
