import numpy as np
from scipy.sparse.linalg import splu

from .jacobian import JacobianStructure
from .active_set import update_active_set, apply_temperature_constraints_sparse
from .residual import assemble_residual_and_jacobian_sparse


def newton_step(
    state,
    physics,
    jac_struct: JacobianStructure,
    A_csc,
    active_T,
    active_Qc,
    limit_T,
    dT_hyst,
    log: bool = False,
):
    """
    Damped Newton iteration with active-set updates and complementarity checks.
    """
    graph = physics.graph
    N = graph.N
    E = graph.E

    maxiter = 500
    alpha = 0.5

    dt = physics.dt
    T_old = physics.T_old

    for it in range(maxiter):
        update_active_set(state, physics, active_T, active_Qc, limit_T, dT_hyst)

        T = state[0:N].copy()
        for i in range(N):
            if active_T[i]:
                T[i] = np.clip(T[i], physics.T_low[i], physics.T_high[i])
        state[0:N] = T

        F, data = assemble_residual_and_jacobian_sparse(state, physics, jac_struct)

        apply_temperature_constraints_sparse(
            state,
            F,
            data,
            jac_struct,
            active_T,
            limit_T,
        )

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

        rhs = -F
        A_csc.data[:] = data
        lu = splu(A_csc)
        delta = lu.solve(rhs)

        state_trial = state + delta
        Qc_trial = state_trial[N + E:N + E + N]

        for i in range(N):
            if not (active_T[i] or active_Qc[i]):
                continue

            low = physics.T_low[i]
            high = physics.T_high[i]

            if np.isclose(limit_T[i], low) and Qc_trial[i] < 0.0:
                active_T[i] = False
                active_Qc[i] = False
                if log:
                    print(f"Releasing node {i} constraint (lower bound, Qc<0)")
                break

            if np.isclose(limit_T[i], high) and Qc_trial[i] > 0.0:
                active_T[i] = False
                active_Qc[i] = False
                if log:
                    print(f"Releasing node {i} constraint (upper bound, Qc>0)")
                break
        else:
            F_trial, _ = assemble_residual_and_jacobian_sparse(
                state_trial, physics, jac_struct
            )
            if np.max(np.abs(F_trial)) < np.max(np.abs(F)):
                state = state_trial
            else:
                state = state + alpha * delta

            if log:
                print(
                    f"it={it}, ||F||={np.max(np.abs(F))}, "
                    f"||delta||={np.max(np.abs(delta))}"
                )

            F_norm = np.max(np.abs(F))
            delta_norm = np.max(np.abs(delta))

            if F_norm < 1e-6:
                return state, it + 1
            if delta_norm * alpha < 1e-10:
                return state, it + 1

            continue

        # constraint released: restart Newton
        continue

    return state, maxiter
