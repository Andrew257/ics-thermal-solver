import numpy as np

from ics_thermal.graph import build_chain_graph
from ics_thermal.physics import PhysicsAssembly
from ics_thermal.active_set import update_active_set


def test_active_set_hysteresis_basic():
    N = 2
    graph = build_chain_graph(N)
    physics = PhysicsAssembly(graph)

    C0 = np.ones(N)
    C1 = np.zeros(N)
    k1 = 0.0
    k2 = 0.0
    dt = 0.1
    T_low = np.array([0.0, 0.0])
    T_high = np.array([1.0, 1.0])
    amp = np.zeros(N)
    period = np.ones(N)
    phase = np.zeros(N)

    physics.set_params(C0, C1, k1, k2, dt, T_low, T_high, amp, period, phase)

    state = np.zeros(graph.n_state)
    state[0:N] = np.array([-0.1, 0.5])  # node 0 below lower bound

    active_T = [False] * N
    active_Qc = [False] * N
    limit_T = np.zeros(N)
    dT_hyst = np.full(N, 0.01)

    update_active_set(state, physics, active_T, active_Qc, limit_T, dT_hyst)

    assert active_T[0] is True
    assert active_Qc[0] is True
    assert limit_T[0] == T_low[0]
    assert active_T[1] is False
