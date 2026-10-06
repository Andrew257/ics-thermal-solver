import numpy as np

from .graph import ICSGraph, build_chain_graph
from .physics import PhysicsAssembly
from .jacobian import JacobianStructure
from .newton import newton_step, reset_newton_stats, newton_summary, newton_stats


def run_ics(
    graph: ICSGraph,
    params,
    t_final: float = 13.0,
    dt_ics: float = 0.05,
    log: bool = False,
):
    """
    General ICS driver on a given graph.

    params = (C0, C1, k1, k2, T_low, T_high, amp, period, phase)
    """
    C0, C1, k1, k2, T_low, T_high, amp, period, phase = params

    physics = PhysicsAssembly(graph)
    physics.set_params(C0, C1, k1, k2, dt_ics, T_low, T_high, amp, period, phase)

    N = graph.N
    E = graph.E
    n = graph.n_state

    jac_struct = JacobianStructure(graph)
    A_csc = jac_struct.empty_csc()

    active_T = [False] * N
    active_Qc = [False] * N
    limit_T = np.zeros(N)

    dT_hyst = np.full(N, 0.001)

    # Reset Newton diagnostics so the counters describe this run only.
    reset_newton_stats()

    # ------------------------------------------------------------
    # State vector: temperatures first, then edge fluxes, then
    # constraint heat flows.  The uniform initial field produces
    # zero conductive flux, so the flux block is seeded to zero and
    # then reconciled by one Newton solve below.
    # ------------------------------------------------------------
    state = np.zeros(n)
    T_init = np.full(N, 50.0)
    state[0:N] = T_init

    for e, (i, j) in enumerate(graph.edges):
        state[graph.idx_Q(e)] = physics.conduction(T_init[i], T_init[j])

    nt = int(round(t_final / dt_ics))

    # ------------------------------------------------------------
    # Initial condition: solve one Newton step from the prescribed
    # T_init so that index 0 is a self-consistent state. Without this
    # the stored fluxes are the trivially-zero conduction of a uniform
    # field, and the balance residual over the first interval compares
    # a genuine flux against a spurious temperature change.
    # ------------------------------------------------------------
    active_T[:] = [False] * N
    active_Qc[:] = [False] * N
    physics.set_time_context(state[0:N], 0.0)

    state, n_iter0 = newton_step(
        state,
        physics,
        jac_struct,
        A_csc,
        active_T,
        active_Qc,
        limit_T,
        dT_hyst,
        log=log,
    )

    times = [0.0]
    T_hist = [state[0:N].copy()]
    Q_hist = [state[N:N + E].copy()]
    Qc_hist = [state[N + E:N + E + N].copy()]

    mode0 = 0
    for i in range(N):
        if active_T[i]:
            mode0 |= (1 << i)
    mode_hist = [mode0]
    n_iter_hist = [n_iter0]

    for itime in range(1, nt):
        t_now = itime * dt_ics
        physics.set_time_context(state[0:N], t_now)

        active_T[:] = [False] * N
        active_Qc[:] = [False] * N

        state, n_iter = newton_step(
            state,
            physics,
            jac_struct,
            A_csc,
            active_T,
            active_Qc,
            limit_T,
            dT_hyst,
            log=log,
        )

        T = state[0:N]
        Q = state[N:N + E]
        Qc = state[N + E:N + E + N]

        mode = 0
        for i in range(N):
            if active_T[i]:
                mode |= (1 << i)

        times.append(t_now)
        T_hist.append(T.copy())
        Q_hist.append(Q.copy())
        Qc_hist.append(Qc.copy())
        mode_hist.append(mode)
        n_iter_hist.append(n_iter)

    if log:
        print("    " + newton_summary())

    return {
        "t": np.array(times),
        "T": np.array(T_hist),
        "Q": np.array(Q_hist),
        "Qc": np.array(Qc_hist),
        "mode": np.array(mode_hist),
        "n_iter": np.array(n_iter_hist),
        "newton_stats": newton_stats(),
    }


def run_ics_chain(
    N: int,
    params,
    t_final: float = 13.0,
    dt_ics: float = 0.05,
    log: bool = False,
):
    graph = build_chain_graph(N)
    return run_ics(graph, params, t_final=t_final, dt_ics=dt_ics, log=log)
