import numpy as np

from .graph import ICSGraph, build_chain_graph


def run_pcs(
    graph: ICSGraph,
    params,
    t_final: float = 13.0,
    h_pcs: float = 1e-5,
):
    """
    Explicit PCS baseline on a general graph.

    params = (C0, C1, k1, k2, T_low, T_high, amp, period, phase)
    """
    C0, C1, k1, k2, T_low, T_high, amp, period, phase = params
    N = graph.N
    E = graph.E

    def C(i, T):
        return C0[i] + C1[i] * T

    def S(i, t):
        if period[i] == 0.0:
            return 0.0
        omega = 2.0 * np.pi / period[i]
        return amp[i] * np.sin(omega * t + phase[i])

    def conduction(Ti, Tj):
        return k1 * (Tj - Ti) + k2 * (Tj**4 - Ti**4)

    def rhs(T, t):
        dT = np.zeros(N)
        Q = np.zeros(E)

        for e, (i, j) in enumerate(graph.edges):
            Q[e] = conduction(T[i], T[j])

        for i in range(N):
            sumQ = 0.0
            for e, sign in graph.node_edges[i]:
                sumQ += sign * Q[e]

            Ci = C(i, T[i])
            Si = S(i, t)

            dT[i] = (-sumQ + Si) / Ci

        return dT, Q

    def compute_mode(T):
        mode = 0
        for i in range(N):
            if T[i] <= T_low[i] or T[i] >= T_high[i]:
                mode |= (1 << i)
        return mode

    t = 0.0
    T = np.full(N, 50.0)

    times = []
    T_hist = []
    Q_hist = []
    Qc_hist = []
    mode_hist = []

    T_prev = None
    icount = 0
    while t < t_final - 1e-12:
        icount += 1
        times.append(t)

        dT, Q = rhs(T, t)
        Q_hist.append(Q.copy())
        T_hist.append(T.copy())

        mode = compute_mode(T)
        mode_hist.append(mode)

        Qc = np.zeros(N)
        if T_prev is not None:
            for i in range(N):
                if mode & (1 << i):
                    sumQ = 0.0
                    for e, sign in graph.node_edges[i]:
                        sumQ += sign * Q[e]

                    Ci = C(i, T[i])
                    Si = S(i, t)

                    Qc[i] = (Ci / h_pcs) * (T[i] - T_prev[i]) + sumQ - Si
        Qc_hist.append(Qc.copy())

        T_prev = T.copy()
        T = T + h_pcs * dT
        T = np.clip(T, T_low, T_high)

        t += h_pcs

    return {
        "t": np.array(times),
        "T": np.array(T_hist),
        "Q": np.array(Q_hist),
        "Qc": np.array(Qc_hist),
        "mode": np.array(mode_hist),
        "icount": icount,
    }


def run_pcs_chain(
    N: int,
    params,
    t_final: float = 13.0,
    h_pcs: float = 1e-5,
):
    graph = build_chain_graph(N)
    return run_pcs(graph, params, t_final=t_final, h_pcs=h_pcs)
