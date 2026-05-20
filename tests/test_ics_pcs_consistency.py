import numpy as np

from ics_thermal.config import run_test_case


def test_ics_pcs_close_for_test_A():
    ics, pcs = run_test_case("A", N=2, solver="both", t_final=1.0, dt_ics=0.05, h_pcs=0.02)

    tI = ics["t"]
    TI = ics["T"]
    tP = pcs["t"]
    TP = pcs["T"]

    # crude alignment: compare ICS temps to PCS temps at closest PCS times
    for k, t in enumerate(tI):
        idx = np.argmin(np.abs(tP - t))
        diff = np.max(np.abs(TI[k] - TP[idx]))
        assert diff < 5.0  # loose tolerance, just sanity check


def test_ics_newton_iterations_positive():
    ics = run_test_case("A", N=2, solver="ICS", t_final=1.0, dt_ics=0.05)
    n_iter = ics["n_iter"]
    assert np.all(n_iter >= 0)
    assert np.any(n_iter > 0)
