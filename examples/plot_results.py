# examples/plot_results.py

import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # Force a GUI backend on Windows
import matplotlib.pyplot as plt


def plot_temperatures(result, title="Temperature trajectories"):
    t = result["t"]
    T = result["T"]
    N = T.shape[1]

    plt.figure(figsize=(8, 4))
    for i in range(N):
        plt.plot(t, T[:, i], label=f"T{i}")
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [°C]")
    plt.title(title)
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()


def plot_heatflows(result, title="Heat flows"):
    t = result["t"]
    Q = result["Q"]
    Qc = result["Qc"]

    E = Q.shape[1]
    N = Qc.shape[1]

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for e in range(E):
        axs[0].plot(t, Q[:, e], label=f"Q{e}")
    axs[0].set_ylabel("Q [W]")
    axs[0].set_title("Conduction–radiation flows")
    axs[0].grid(True)
    axs[0].legend()

    for i in range(N):
        axs[1].plot(t, Qc[:, i], label=f"Qc{i}")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Qc [W]")
    axs[1].set_title("Constraint heat flows")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()


def plot_modes(result, title="Mode timeline"):
    t = result["t"]
    mode = result["mode"]

    plt.figure(figsize=(8, 3))
    plt.step(t, mode, where="post")
    plt.xlabel("Time [s]")
    plt.ylabel("Mode")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()


def plot_newton_iterations(result, title="Newton iterations per macrostep"):
    t = result["t"]
    n_iter = result["n_iter"]

    plt.figure(figsize=(8, 3))
    plt.step(t, n_iter, where="post")
    plt.xlabel("Time [s]")
    plt.ylabel("Iterations")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()


def plot_compare_ics_pcs(ics, pcs, nodes=None):
    # Extract ICS data
    tI   = ics["t"]
    TI   = ics["T"]
    QI   = ics["Q"]
    QcI  = ics["Qc"]
    modeI = ics["mode"]

    # Extract PCS data
    tP   = pcs["t"]
    TP   = pcs["T"]
    QP   = pcs["Q"]
    QcP  = pcs["Qc"]
    modeP = pcs["mode"]

    N = TI.shape[1]      # total number of nodes
    E = QI.shape[1]      # number of edges

    # ------------------------------------------------------------
    # Select nodes
    # ------------------------------------------------------------
    if nodes is None:
        nodes = list(range(N))
    else:
        nodes = sorted([i for i in nodes if 0 <= i < N])

    # ============================================================
    # 1. Temperatures
    # ============================================================
    plt.figure(figsize=(10, 4))
    for i in nodes:
        plt.plot(tI, TI[:, i], label=f"ICS T{i}", linewidth = 2.5)
        plt.plot(tP, TP[:, i], "--", label=f"PCS T{i}", linewidth = 2.5)
    plt.xlabel("Time t [s]")
    plt.ylabel("Temperature T [°C]")
    plt.grid(True)
    plt.legend(loc="upper right", ncol=2)
    plt.tight_layout()

    # ============================================================
    # 2. Heat flows (edges)
    # Only plot edges adjacent to selected nodes
    # ============================================================
    edges_to_plot = sorted(set(
        e for i in nodes for e in [i-1, i] if 0 <= e < E
    ))

    plt.figure(figsize=(10, 4))
    for e in edges_to_plot:
        plt.plot(tI, QI[:, e], label=f"ICS Q{e}", linewidth = 2.5)
        plt.plot(tP, QP[:, e], "--", label=f"PCS Q{e}", linewidth = 2.5)
    plt.xlabel("Time t [s]")
    plt.ylabel("Heat flow Q [W]")
    plt.grid(True)
    plt.legend(loc="upper right", ncol=2)
    plt.tight_layout()

    # ============================================================
    # 3. Constraint flows (per node)
    # ============================================================
    plt.figure(figsize=(10, 4))
    for i in nodes:
        plt.plot(tI, QcI[:, i], label=f"ICS Qc{i}", linewidth = 2.5)
        plt.plot(tP, QcP[:, i], "--", label=f"PCS Qc{i}", linewidth = 2.5)
    plt.xlabel("Time t [s]")
    plt.ylabel("Constraint flow Qc [W]")
    plt.grid(True)
    plt.legend(loc="upper right", ncol=2)
    plt.tight_layout()

    # ============================================================
    # 4. Constraint-activation timeline (bitmask)
    # Only show selected nodes
    # ============================================================
    plt.figure(figsize=(10, 3))

    # Decode bitmask for ICS
    modeI_sel = np.zeros_like(modeI)
    for i in nodes:
        modeI_sel |= ((modeI >> i) & 1) << i

    # Decode bitmask for PCS
    modeP_sel = np.zeros_like(modeP)
    for i in nodes:
        modeP_sel |= ((modeP >> i) & 1) << i

    plt.step(tI, modeI_sel, where="post", label="ICS")
    plt.step(tP, modeP_sel, where="post", linestyle="--", label="PCS")

    plt.xlabel("Time t [s]")
    plt.ylabel("Mode (bitmask)")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()

def plot_phase_planes_ics_pcs(ics, pcs):
    # --- Extract data ---
    tI   = ics["t"]
    TI   = ics["T"]      # (nI, N_I)
    QcI  = ics["Qc"]     # (nI, N_I)
    modeI = ics["mode"]  # (nI,)

    tP   = pcs["t"]
    TP   = pcs["T"]      # (nP, N_P)
    QcP  = pcs["Qc"]     # (nP, N_P)
    modeP = pcs["mode"]  # (nP,)

    # --- Determine common number of nodes ---
    N_I = TI.shape[1]
    N_P = TP.shape[1]
    N   = min(N_I, N_P)  # only plot nodes that exist in both

    # ============================================================
    # 1. Phase planes (T_i vs Qc_i) for each node, N-safe layout
    # ============================================================

    # Automatic subplot grid: as square as possible
    nrows = int(np.floor(np.sqrt(N)))
    ncols = int(np.ceil(N / nrows))

    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)
    # fig.suptitle("Phase planes: (T_i, Qc_i) — ICS vs PCS")

    for i in range(N):
        r = i // ncols
        c = i % ncols
        ax = axs[r, c]

        ax.scatter(TI[:, i], QcI[:, i],
                   facecolors='none', edgecolors='blue', s=40,
                   label=f"ICS node {i}")
        ax.scatter(TP[:, i], QcP[:, i],
                   marker=".", s=15, color="orange",
                   label=f"PCS node {i}")

        ax.set_xlabel(f"T{i} [°C]")
        ax.set_ylabel(f"Qc{i} [W]")
        ax.grid(True)
        ax.legend(fontsize=8, loc="upper right")

    # Hide any unused subplots (if N < nrows*ncols)
    for j in range(N, nrows*ncols):
        r = j // ncols
        c = j % ncols
        axs[r, c].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ============================================================
    # 2. Heatmap of constraint activation over time (ICS & PCS)
    # ============================================================

    # Decode mode bitmasks into (time, node) boolean arrays
    def decode_modes(mode_array, Nnodes):
        nT = len(mode_array)
        active = np.zeros((nT, Nnodes), dtype=int)
        for k in range(nT):
            m = int(mode_array[k])
            for i in range(Nnodes):
                if m & (1 << i):
                    active[k, i] = 1
        return active

    activeI = decode_modes(modeI, N_I)
    activeP = decode_modes(modeP, N_P)

    # We plot up to N nodes for each, aligned by node index
    N_heat = max(N_I, N_P)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    # fig.suptitle("Constraint activation heatmap (1 = constrained)")

    # ICS heatmap
    ax = axs[0]
    im = ax.imshow(activeI.T, aspect="auto", origin="lower",
                   extent=[tI[0], tI[-1], -0.5, N_I-0.5],
                   cmap="Reds",
                   interpolation="nearest",   # <- key line
    )
    ax.set_ylabel("Node index")
    ax.set_title("ICS constraint activation")
    ax.set_yticks(range(N_I))
    fig.colorbar(im, ax=ax, label="Active")

    # PCS heatmap
    ax = axs[1]
    # im = ax.imshow(activeP.T, aspect="auto", origin="lower",
    #                extent=[tP[0], tP[-1], -0.5, N_P-0.5],
    #                cmap="Reds")
    im = ax.imshow(
    activeP.T,
    aspect="auto",
    origin="lower",
    extent=[tP[0], tP[-1], -0.5, N_P - 0.5],
    cmap="Reds",
    interpolation="nearest",   # <- key line
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Node index")
    ax.set_title("PCS constraint activation")
    ax.set_yticks(range(N_P))
    fig.colorbar(im, ax=ax, label="Active")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])