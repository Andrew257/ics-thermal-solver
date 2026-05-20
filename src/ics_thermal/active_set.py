import numpy as np

from .jacobian import JacobianStructure


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
