import numpy as np
from scipy.sparse import csc_matrix

from .graph import ICSGraph


class JacobianStructure:
    """
    Symbolic sparsity pattern and CSC structure for the ICS Jacobian.

    State ordering:
        T[0..N-1]
        Q[0..E-1]
        Qc[0..N-1]

    Rows:
        T rows: node energy balances
        Q rows: edge conduction equations
        Qc rows: constraint equations (fully active pattern)
    """

    def __init__(self, graph: ICSGraph):
        self.graph = graph
        self.N = graph.N
        self.E = graph.E
        self.n = graph.n_state

        pattern = np.zeros((self.n, self.n), dtype=bool)

        # T rows
        for i in range(self.N):
            row = graph.idx_T(i)
            pattern[row, graph.idx_T(i)] = True
            for e, _sign in graph.node_edges[i]:
                pattern[row, graph.idx_Q(e)] = True
            pattern[row, graph.idx_Qc(i)] = True

        # Q rows
        for e, (i, j) in enumerate(graph.edges):
            row = graph.idx_Q(e)
            pattern[row, graph.idx_T(i)] = True
            pattern[row, graph.idx_T(j)] = True
            pattern[row, graph.idx_Q(e)] = True

        # Qc rows (fully active)
        for i in range(self.N):
            row = graph.idx_Qc(i)
            pattern[row, graph.idx_T(i)] = True
            for e, _sign in graph.node_edges[i]:
                pattern[row, graph.idx_Q(e)] = True
            pattern[row, graph.idx_Qc(i)] = True

        self.pattern = pattern
        self._build_csc_structure()

    def _build_csc_structure(self):
        n = self.n
        pattern = self.pattern

        col_nnz = np.zeros(n, dtype=int)
        for j in range(n):
            col_nnz[j] = np.count_nonzero(pattern[:, j])

        indptr = np.zeros(n + 1, dtype=int)
        indptr[1:] = np.cumsum(col_nnz)

        nnz = indptr[-1]
        indices = np.zeros(nnz, dtype=int)

        pos = indptr.copy()
        for j in range(n):
            rows = np.nonzero(pattern[:, j])[0]
            for r in rows:
                k = pos[j]
                indices[k] = r
                pos[j] += 1

        lookup = {}
        for j in range(n):
            for k in range(indptr[j], indptr[j + 1]):
                r = indices[k]
                lookup[(r, j)] = k

        self.indptr = indptr
        self.indices = indices
        self.nnz = nnz
        self.lookup = lookup

        row_cols = [[] for _ in range(n)]
        for (r, c), _k in lookup.items():
            row_cols[r].append(c)
        self.row_cols = row_cols

    def empty_csc(self) -> csc_matrix:
        data = np.zeros(self.nnz, dtype=float)
        return csc_matrix((data, self.indices, self.indptr), shape=(self.n, self.n))

    def new_data(self):
        return np.zeros(self.nnz, dtype=float)

    def set_entry(self, data, row, col, value):
        k = self.lookup.get((row, col), None)
        if k is not None:
            data[k] = value

    def add_entry(self, data, row, col, value):
        k = self.lookup.get((row, col), None)
        if k is not None:
            data[k] += value

    def to_csc(self, data):
        return csc_matrix((data, self.indices, self.indptr), shape=(self.n, self.n))
