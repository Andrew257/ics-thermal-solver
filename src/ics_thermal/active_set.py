import numpy as np

from .jacobian import JacobianStructure


def apply_constraint_flow_rows(state, F, data, jac_struct, physics, active_Qc):
    """
    Overwrite the Qc rows for active constraints with the
    constraint-consistent condition F = Qc_i - target_Qc_i.
    """
    graph = physics.graph
    N = graph.N
    E = graph.E
    dt = physics.dt
    T_old = physics.T_old

    T = state[0:N]
    Q = state[N:N + E]
    Qc = state[N + E:N + E + N]

    for i in range(N):
        if not active_Qc[i]:
            continue

        Ti = T[i]
        Ci = physics.C(i, Ti)
        dCi_dTi = physics.C1[i]
        Si = physics.S(i)

        sumQ = 0.0
        for e, sign in graph.node_edges[i]:
            sumQ += sign * Q[e]

        target_Qc = (Ci / dt) * (Ti - T_old[i]) + sumQ - Si

        row = graph.idx_Qc(i)
        F[row] = Qc[i] - target_Qc

        dtarget_dTi = (dCi_dTi / dt) * (Ti - T_old[i]) + Ci / dt
        jac_struct.set_entry(data, row, graph.idx_T(i), -dtarget_dTi)
        for e, sign in graph.node_edges[i]:
            jac_struct.set_entry(data, row, graph.idx_Q(e), -sign)
        jac_struct.set_entry(data, row, graph.idx_Qc(i), 1.0)


def update_active_set(
    state,
    physics,
    active_T,
    active_Qc,
    limit_T,
    dT_hyst,
):
    """
    Hysteresis-based active set update for each node i.
    (Ti, Qc_i) activate/deactivate together.
    """
    N = physics.graph.N
    T = state[0:N]

    T_low = physics.T_low
    T_high = physics.T_high

    for i in range(N):
        Ti = T[i]
        low = T_low[i]
        high = T_high[i]
        dT = dT_hyst[i]

        if active_T[i] or active_Qc[i]:
            if (Ti > low + dT) and (Ti < high - dT):
                active_T[i] = False
                active_Qc[i] = False
        else:
            if Ti < low - dT:
                active_T[i] = True
                active_Qc[i] = True
                limit_T[i] = low
            elif Ti > high + dT:
                active_T[i] = True
                active_Qc[i] = True
                limit_T[i] = high

        if not active_T[i]:
            limit_T[i] = Ti


def apply_temperature_constraints_sparse(
    state,
    F,
    data,
    jac_struct: JacobianStructure,
    active_T,
    limit_T,
):
    """
    Overwrite T rows to enforce active temperature constraints.
    Works directly on CSC data via row-wise column lists.
    """
    N = len(active_T)
    row_cols = jac_struct.row_cols

    for i in range(N):
        if not active_T[i]:
            continue

        row = i
        for col in row_cols[row]:
            jac_struct.set_entry(data, row, col, 0.0)

        jac_struct.set_entry(data, row, row, 1.0)
        F[row] = state[row] - limit_T[i]
