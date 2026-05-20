import numpy as np


class ICSGraph:
    """
    General graph topology for ICS.

    Nodes: 0..N-1
    Edges: list of (i, j) with orientation i -> j
           Node i sees -Q_e, node j sees +Q_e in its energy balance.
    """

    def __init__(self, N, edges):
        self.N = int(N)
        self.edges = [(int(i), int(j)) for (i, j) in edges]
        self.E = len(self.edges)

        # Incidence lists: for each node, list of (edge_index, sign)
        # sign = -1 if node is tail (i), +1 if node is head (j)
        self.node_edges = [[] for _ in range(self.N)]
        for e, (i, j) in enumerate(self.edges):
            self.node_edges[i].append((e, -1.0))  # outflow
            self.node_edges[j].append((e, +1.0))  # inflow

    def idx_T(self, i: int) -> int:
        return i

    def idx_Q(self, e: int) -> int:
        return self.N + e

    def idx_Qc(self, i: int) -> int:
        return self.N + self.E + i

    @property
    def n_state(self) -> int:
        return 2 * self.N + self.E


def build_chain_graph(N: int) -> ICSGraph:
    """
    Convenience: 1-D chain graph with edges (0,1), (1,2), ..., (N-2,N-1).
    Orientation: i -> i+1, consistent with the chain sign convention.
    """
    edges = [(i, i + 1) for i in range(N - 1)]
    return ICSGraph(N, edges)
