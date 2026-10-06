import numpy as np
from scipy.sparse.linalg import splu

from .jacobian import JacobianStructure
from .active_set import (
    update_active_set,
    apply_temperature_constraints_sparse,
    apply_constraint_flow_rows,
)
from .residual import assemble_residual_and_jacobian_sparse


# ---------------------------------------------------------------
# Global counters for Newton diagnostics.
# Reset with reset_newton_stats() before a run, read with
# newton_stats() afterwards.
# ---------------------------------------------------------------
_NEWTON_STATS = {
    "iterations": 0,
    "full_steps": 0,
    "damped_steps": 0,
    "releases": 0,
    "stalls": 0,
    "not_converged": 0,
}


def reset_newton_stats():
    """Zero the diagnostic counters before a run."""
    for k in _NEWTON_STATS:
        _NEWTON_STATS[k] = 0


def newton_stats():
    """Return a copy of the diagnostic counters."""
    return dict(_NEWTON_STATS)


def newton_summary():
    """Human-readable one-line summary.

    The percentage is taken over the steps actually taken, not over the
    iteration count: a released iteration takes no step at all and must
    not enter the denominator.
    """
    s = _NEWTON_STATS
    steps = s["full_steps"] + s["damped_steps"]
    if steps == 0:
        return "no Newton steps recorded"
    pct_full = 100.0 * s["full_steps"] / steps
    return (f"Newton iterations: {s['iterations']}  |  "
            f"steps taken: {steps}  |  "
            f"full (alpha=1): {s['full_steps']} ({pct_full:.2f}%)  |  "
            f"backtracked: {s['damped_steps']} ({100.0 - pct_full:.2f}%)  |  "
            f"releases: {s['releases']}  |  "
            f"stalls: {s['stalls']}  |  "
            f"not converged: {s['not_converged']}")


def _constrained_residual(state, physics, jac_struct, data,
                          active_T, active_Qc, limit_T):
    """
    Assemble the residual with BOTH constraint overwrites applied:

      - temperature rows of active nodes replaced by  T_i - limit_i
      - constraint-flow rows of active nodes replaced by
            Qc_i - target_Qc_i

    Both are required. Without the second, the stored multiplier enters
    the norm directly and the residual can never fall below tolerance.
    """
    F = assemble_residual_and_jacobian_sparse.__wrapped__(  # see note below
        state, physics, jac_struct
    )[0] if hasattr(assemble_residual_and_jacobian_sparse, "__wrapped__") \
        else assemble_residual_and_jacobian_sparse(state, physics, jac_struct)[0]

    apply_temperature_constraints_sparse(
        state, F, data, jac_struct, active_T, limit_T
    )
    apply_constraint_flow_rows(
        state, F, data, jac_struct, physics, active_Qc
    )
    return F


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
    Damped Newton iteration with active-set updates, backtracking line
    search and complementarity checks.

    Returns (state, n_iter).  n_iter < maxiter means the residual
    criterion was satisfied on the fully constrained system.  If the
    budget is exhausted, the best iterate seen is returned with
    n_iter == maxiter and 'not_converged' incremented.
    """
    graph = physics.graph
    N = graph.N
    E = graph.E

    maxiter = 500
    alpha = 0.5
    max_backtrack = 8          # 0.5 .. 0.5^8

    dt = physics.dt
    T_old = physics.T_old

    best_state = state.copy()
    best_norm = np.inf

    for it in range(maxiter):
        update_active_set(state, physics, active_T, active_Qc, limit_T, dT_hyst)

        T = state[0:N].copy()
        for i in range(N):
            if active_T[i]:
                T[i] = np.clip(T[i], physics.T_low[i], physics.T_high[i])
        state[0:N] = T

        # ---- assemble the full constrained system -----------------
        F, data = assemble_residual_and_jacobian_sparse(state, physics, jac_struct)

        apply_temperature_constraints_sparse(
            state, F, data, jac_struct, active_T, limit_T
        )
        apply_constraint_flow_rows(
            state, F, data, jac_struct, physics, active_Qc
        )

        F_norm = np.max(np.abs(F))
        if F_norm < best_norm:
            best_norm = F_norm
            best_state = state.copy()

        # ---- solve the linear system ------------------------------
        rhs = -F
        A_csc.data[:] = data
        lu = splu(A_csc)
        delta = lu.solve(rhs)

        state_trial = state + delta
        Qc_trial = state_trial[N + E:N + E + N]

        _NEWTON_STATS["iterations"] += 1

        # ---- complementarity check -------------------------------
        released = False
        for i in range(N):
            if not (active_T[i] or active_Qc[i]):
                continue

            low = physics.T_low[i]
            high = physics.T_high[i]

            if np.isclose(limit_T[i], low) and Qc_trial[i] < 0.0:
                active_T[i] = False
                active_Qc[i] = False
                _NEWTON_STATS["releases"] += 1
                if log:
                    print(f"Releasing node {i} constraint (lower bound, Qc<0)")
                released = True
                break

            if np.isclose(limit_T[i], high) and Qc_trial[i] > 0.0:
                active_T[i] = False
                active_Qc[i] = False
                _NEWTON_STATS["releases"] += 1
                if log:
                    print(f"Releasing node {i} constraint (upper bound, Qc>0)")
                released = True
                break

        if released:
            continue

        # ---- backtracking line search -----------------------------
        # Both overwrites must be applied to every trial residual, so
        # that each candidate is compared against the same equations.
        accepted = False

        F_trial, data_trial = assemble_residual_and_jacobian_sparse(
            state_trial, physics, jac_struct
        )
        apply_temperature_constraints_sparse(
            state_trial, F_trial, data_trial, jac_struct, active_T, limit_T
        )
        apply_constraint_flow_rows(
            state_trial, F_trial, data_trial, jac_struct, physics, active_Qc
        )

        if np.max(np.abs(F_trial)) < F_norm:
            state = state_trial
            _NEWTON_STATS["full_steps"] += 1
            accepted = True
        else:
            for bt in range(max_backtrack):
                alpha_bt = alpha ** (bt + 1)
                trial_bt = state + alpha_bt * delta

                F_bt, data_bt = assemble_residual_and_jacobian_sparse(
                    trial_bt, physics, jac_struct
                )
                apply_temperature_constraints_sparse(
                    trial_bt, F_bt, data_bt, jac_struct, active_T, limit_T
                )
                apply_constraint_flow_rows(
                    trial_bt, F_bt, data_bt, jac_struct, physics, active_Qc
                )

                if np.max(np.abs(F_bt)) < F_norm:
                    state = trial_bt
                    _NEWTON_STATS["damped_steps"] += 1
                    accepted = True
                    break

        if not accepted:
            _NEWTON_STATS["stalls"] += 1
            if log:
                print(f"it={it}: line search exhausted, ||F||={F_norm:.3e}")
            state = state + (alpha ** max_backtrack) * delta

        # ---- convergence test on the fully constrained residual ---
        F_check, data_check = assemble_residual_and_jacobian_sparse(
            state, physics, jac_struct
        )
        apply_temperature_constraints_sparse(
            state, F_check, data_check, jac_struct, active_T, limit_T
        )
        apply_constraint_flow_rows(
            state, F_check, data_check, jac_struct, physics, active_Qc
        )
        F_check_norm = np.max(np.abs(F_check))

        if F_check_norm < best_norm:
            best_norm = F_check_norm
            best_state = state.copy()

        if log:
            print(f"it={it}, ||F||={F_check_norm:.3e}, "
                  f"||delta||={np.max(np.abs(delta)):.3e}")

        delta_norm = np.max(np.abs(delta))

        if F_check_norm < 1e-6:
            return state, it + 1
        if delta_norm < 1e-10:
            return state, it + 1

    _NEWTON_STATS["not_converged"] += 1
    if log:
        print(f"WARNING: Newton did not converge in {maxiter} iterations; "
              f"best ||F|| = {best_norm:.3e}")
    return best_state, maxiter
