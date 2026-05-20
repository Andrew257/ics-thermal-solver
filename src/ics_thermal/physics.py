import numpy as np

from .graph import ICSGraph


class PhysicsAssembly:
    """
    Physics for ICS on a general graph.

    - Temperature-dependent capacities C_i(Ti) = C0[i] + C1[i]*Ti
    - Conduction + radiation along edges
    - Internal heat generation S_i(t)
    - Per-node temperature limits (for constraints)
    """

    def __init__(self, graph: ICSGraph):
        self.graph = graph
        N = graph.N

        self.C0 = np.zeros(N)
        self.C1 = np.zeros(N)

        self.k1 = 0.0
        self.k2 = 0.0

        self.dt = 0.0

        self.amp = np.zeros(N)
        self.period = np.ones(N)
        self.phase = np.zeros(N)

        self.T_low = np.zeros(N)
        self.T_high = np.zeros(N)

        self.T_old = np.zeros(N)
        self.t_now = 0.0

    def set_params(
        self,
        C0,
        C1,
        k1,
        k2,
        dt,
        T_low,
        T_high,
        amp,
        period,
        phase=None,
    ):
        self.C0 = np.array(C0, dtype=float)
        self.C1 = np.array(C1, dtype=float)

        self.k1 = float(k1)
        self.k2 = float(k2)
        self.dt = float(dt)

        self.T_low = np.array(T_low, dtype=float)
        self.T_high = np.array(T_high, dtype=float)

        self.amp = np.array(amp, dtype=float)
        self.period = np.array(period, dtype=float)
        if phase is None:
            self.phase = np.zeros_like(self.amp)
        else:
            self.phase = np.array(phase, dtype=float)

    def set_time_context(self, T_old, t_now: float):
        self.T_old = np.array(T_old, dtype=float)
        self.t_now = float(t_now)

    def C(self, i: int, Ti: float) -> float:
        return self.C0[i] + self.C1[i] * Ti

    def S(self, i: int) -> float:
        if self.period[i] == 0.0:
            return 0.0
        omega = 2.0 * np.pi / self.period[i]
        return self.amp[i] * np.sin(omega * self.t_now + self.phase[i])

    def conduction(self, Ti: float, Tj: float) -> float:
        return self.k1 * (Tj - Ti) + self.k2 * (Tj**4 - Ti**4)
