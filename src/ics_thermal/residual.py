import numpy as np

from .jacobian import JacobianStructure
from .physics import PhysicsAssembly


def assemble_residual_and_jacobian_sparse(
    state,
    physics: PhysicsAssembly,
    jac_struct: JacobianStructure,
):
    """
    Assemble residual F(state) and sparse Jacobian data array
    for the unconstrained system (before T-row overwrites and
    Qc active equations).
    """
    graph = physics.graph
    N = graph.N
    E = graph.E
    n = graph.n_state

    T = state[0:N]
    Q = state[N:N + E]
    Qc = state[N + E:N + E + N]

    dt = physics.dt
    T_old = physics.T_old

    F = np.zeros(n)
    data = jac_struct.new_data()

    # T rows: node energy balances
    for i in range(N):
        Ti = T[i]
        Ci = physics.C(i, Ti)
        dCi_dTi = physics.C1[i]
        Si = physics.S(i)

        sumQ = 0.0
        for e, sign in graph.node_edges[i]:
            sumQ += sign * Q[e]

        row = graph.idx_T(i)

        F[row] = Ci / dt * (Ti - T_old[i]) + sumQ - Qc[i] - Si

        dF_dTi = (Ci / dt) + (dCi_dTi / dt) * (Ti - T_old[i])
        jac_struct.set_entry(data, row, graph.idx_T(i), dF_dTi)

        for e, sign in graph.node_edges[i]:
            jac_struct.set_entry(data, row, graph.idx_Q(e), sign)

        jac_struct.set_entry(data, row, graph.idx_Qc(i), -1.0)

    # Q rows: edge conduction + radiation
    for e, (i, j) in enumerate(graph.edges):
        Ti = T[i]
        Tj = T[j]

        Q_expr = physics.conduction(Ti, Tj)
        row = graph.idx_Q(e)

        F[row] = Q[e] - Q_expr

        dQdTi = -physics.k1 - 4.0 * physics.k2 * Ti**3
        dQdTj = physics.k1 + 4.0 * physics.k2 * Tj**3

        jac_struct.set_entry(data, row, graph.idx_T(i), -dQdTi)
        jac_struct.set_entry(data, row, graph.idx_T(j), -dQdTj)
        jac_struct.set_entry(data, row, graph.idx_Q(e), 1.0)

    # Qc rows: default Qc[i] = 0 (will be overwritten when active)
    for i in range(N):
        row = graph.idx_Qc(i)
        F[row] = Qc[i]
        jac_struct.set_entry(data, row, graph.idx_Qc(i), 1.0)

    return F, data
