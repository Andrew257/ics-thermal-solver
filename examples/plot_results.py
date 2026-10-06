# examples/plot_results.py

import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")  # Force GUI backend if available
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib import ticker

# Distinguishable line styles for multi-node / multi-edge comparisons
NODE_STYLES = [
    {"linestyle": "-",  "dashes": ()},          # Solid
    {"linestyle": "--", "dashes": (6, 3)},       # Dashed
    {"linestyle": "-.", "dashes": (4, 2, 1, 2)}, # Dash-dot
    {"linestyle": ":",  "dashes": (1.5, 2)},     # Dotted
    {"linestyle": "-",  "dashes": (8, 2, 2, 2)}, # Long-dash short-dash
    {"linestyle": "--", "dashes": (3, 2)},       # Short-dash
]

# High-contrast color cycle (converts to distinct, non-clashing gray values)
COLOR_CYCLE = [
    "#000000",  # Black
    "#004488",  # Dark Blue (converts to distinct dark gray)
    "#BB5566",  # Dark Red/Coral (converts to distinct mid gray)
    "#DDAA33",  # Gold/Olive (converts to distinct light-mid gray)
    "#117733",  # Deep Green (converts to distinct mid gray)
    "#666666",  # Medium Gray
]


def plot_temperatures(result, title="Temperature trajectories"):
    t = result["t"]
    T = result["T"]
    N = T.shape[1]

    plt.figure(figsize=(8, 4))
    for i in range(N):
        style = NODE_STYLES[i % len(NODE_STYLES)]
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        plt.plot(t, T[:, i], label=f"T{i}",
                 color=color, linestyle=style["linestyle"], linewidth=1.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [°C]")
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="center right", framealpha=0.95)
    plt.tight_layout()


def plot_heatflows(result, title="Heat flows"):
    t = result["t"]
    Q = result["Q"]
    Qc = result["Qc"]

    E = Q.shape[1]
    N = Qc.shape[1]

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for e in range(E):
        style = NODE_STYLES[e % len(NODE_STYLES)]
        color = COLOR_CYCLE[e % len(COLOR_CYCLE)]
        axs[0].plot(t, Q[:, e], label=f"Q{e}",
                    color=color, linestyle=style["linestyle"], linewidth=1.8)
    axs[0].set_ylabel("Q [W]")
    axs[0].set_title("Conduction–radiation flows")
    axs[0].grid(True, linestyle=":", alpha=0.6)
    axs[0].legend(loc="upper right", framealpha=0.95)

    for i in range(N):
        style = NODE_STYLES[i % len(NODE_STYLES)]
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        axs[1].plot(t, Qc[:, i], label=f"Qc{i}",
                    color=color, linestyle=style["linestyle"], linewidth=1.8)
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Qc [W]")
    axs[1].set_title("Constraint heat flows")
    axs[1].grid(True, linestyle=":", alpha=0.6)
    axs[1].legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()


def plot_modes(result, title="Mode timeline"):
    t = result["t"]
    mode = result["mode"]

    plt.figure(figsize=(8, 3))
    plt.step(t, mode, where="post", color="black", linewidth=1.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Mode")
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()


def plot_newton_iterations(result, title="Newton iterations per macrostep"):
    t = result["t"]
    n_iter = result["n_iter"]

    plt.figure(figsize=(8, 3))
    plt.step(t, n_iter, where="post", color="black", linewidth=1.5)
    plt.xlabel("Time [s]")
    plt.ylabel("Iterations")

    # Iteration counts are integers; force integer tick labels.
    ax = plt.gca()
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()


def plot_compare_ics_pcs(ics, pcs, nodes=None):
    """
    Comparison between ICS and PCS:
      - ICS is rendered as a solid line.
      - PCS is rendered as a distinct dashed line.
      - Contrast holds whether viewed in color online or printed in grayscale.
    """
    tI    = ics["t"]
    TI    = ics["T"]
    QI    = ics["Q"]
    QcI   = ics["Qc"]
    modeI = ics["mode"]

    tP    = pcs["t"]
    TP    = pcs["T"]
    QP    = pcs["Q"]
    QcP   = pcs["Qc"]
    modeP = pcs["mode"]

    N = TI.shape[1]
    E = QI.shape[1]

    if nodes is None:
        nodes = list(range(N))
    else:
        nodes = sorted([i for i in nodes if 0 <= i < N])

    # ============================================================
    # 1. Temperatures
    # ============================================================
    plt.figure(figsize=(9, 4.2))
    for idx, i in enumerate(nodes):
        color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
        # ICS: Solid stroke
        plt.plot(tI, TI[:, i],
                 color=color, linestyle="-", linewidth=2.0,
                 label=f"ICS T{i}")
        # PCS: Dashed stroke
        plt.plot(tP, TP[:, i],
                 color=color, linestyle=(0, (4, 3)), linewidth=1.6, alpha=0.9,
                 label=f"PCS T{i} (ref)")
    plt.xlabel("Time t [s]")
    plt.ylabel("Temperature T [°C]")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="center right", ncol=min(4, len(nodes)), framealpha=0.95, fontsize=9)
    plt.tight_layout()

    # ============================================================
    # 2. Heat flows (edges)
    # ============================================================
    edges_to_plot = sorted(set(
        e for i in nodes for e in [i-1, i] if 0 <= e < E
    ))

    plt.figure(figsize=(9, 4.2))
    for idx, e in enumerate(edges_to_plot):
        color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
        plt.plot(tI, QI[:, e],
                 color=color, linestyle="-", linewidth=2.0,
                 label=f"ICS Q{e}")
        plt.plot(tP, QP[:, e],
                 color=color, linestyle=(0, (4, 3)), linewidth=1.6, alpha=0.9,
                 label=f"PCS Q{e} (ref)")
    plt.xlabel("Time t [s]")
    plt.ylabel("Heat flow Q [W]")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", ncol=min(4, len(edges_to_plot)), framealpha=0.95, fontsize=9)
    plt.tight_layout()

    # ============================================================
    # 3. Constraint flows (per node)
    # ============================================================
    plt.figure(figsize=(9, 4.2))
    for idx, i in enumerate(nodes):
        color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
        plt.plot(tI, QcI[:, i],
                 color=color, linestyle="-", linewidth=2.0,
                 label=f"ICS Qc{i}")
        plt.plot(tP, QcP[:, i],
                 color=color, linestyle=(0, (4, 3)), linewidth=1.6, alpha=0.9,
                 label=f"PCS Qc{i} (ref)")
    plt.xlabel("Time t [s]")
    plt.ylabel("Constraint flow Qc [W]")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", ncol=min(4, len(nodes)), framealpha=0.95, fontsize=9)
    plt.tight_layout()

    # ============================================================
    # 4. Constraint-activation timeline (bitmask)
    # ============================================================
    plt.figure(figsize=(9, 3.2))

    modeI_sel = np.zeros_like(modeI)
    for i in nodes:
        modeI_sel |= ((modeI >> i) & 1) << i

    modeP_sel = np.zeros_like(modeP)
    for i in nodes:
        modeP_sel |= ((modeP >> i) & 1) << i

    plt.step(tI, modeI_sel, where="post", color="black", linestyle="-", linewidth=2.0, label="ICS")
    plt.step(tP, modeP_sel, where="post", color="#555555", linestyle=(0, (4, 2)), linewidth=1.5, label="PCS (ref)")

    plt.xlabel("Time t [s]")
    plt.ylabel("Mode (bitmask)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.95)
    plt.tight_layout()


def plot_phase_planes_ics_pcs(ics, pcs):
    """
    Phase planes and activation heatmaps:
      - Phase planes: ICS = hollow circles with black border; PCS = filled gray dots.
      - Heatmap: Monochromatic 'Greys' map (Black = Active, White = Inactive).
    """
    tI    = ics["t"]
    TI    = ics["T"]
    QcI   = ics["Qc"]
    modeI = ics["mode"]

    tP    = pcs["t"]
    TP    = pcs["T"]
    QcP   = pcs["Qc"]
    modeP = pcs["mode"]

    N_I = TI.shape[1]
    N_P = TP.shape[1]
    N   = min(N_I, N_P)

    # ============================================================
    # 1. Phase planes (T_i vs Qc_i)
    # ============================================================
    nrows = int(np.floor(np.sqrt(N)))
    ncols = int(np.ceil(N / nrows))

    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.8*nrows), squeeze=False)

    for i in range(N):
        r = i // ncols
        c = i % ncols
        ax = axs[r, c]

        # ICS: Hollow circles with crisp black edge
        ax.scatter(TI[:, i], QcI[:, i],
                   facecolors='none', edgecolors='black', linewidths=1.2, s=35,
                   label=f"ICS node {i}")
        # PCS: Small filled gray dots
        ax.scatter(TP[:, i], QcP[:, i],
                   marker=".", s=18, color="#555555", alpha=0.8,
                   label=f"PCS node {i}")

        ax.set_xlabel(f"T{i} [°C]")
        ax.set_ylabel(f"Qc{i} [W]")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.95)

    for j in range(N, nrows*ncols):
        r = j // ncols
        c = j % ncols
        axs[r, c].axis("off")

    plt.tight_layout()

    # ============================================================
    # 2. Heatmap of constraint activation over time (ICS & PCS)
    # ============================================================
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

    fig, axs = plt.subplots(2, 1, figsize=(9.5, 5.5), sharex=True)

    # 'Greys' map: 0 = White (Off), 1 = Solid Black (On)
    ax = axs[0]
    im0 = ax.imshow(activeI.T, aspect="auto", origin="lower",
                    extent=[tI[0], tI[-1], -0.5, N_I-0.5],
                    cmap="Greys", vmin=0, vmax=1,
                    interpolation="nearest")
    ax.set_ylabel("Node index")
    ax.set_title("ICS constraint activation (Black = Active, White = Inactive)", fontsize=10)
    ax.set_yticks(range(N_I))
    cbar0 = fig.colorbar(im0, ax=ax, ticks=[0, 1])
    cbar0.ax.set_yticklabels(["Off", "On"])

    ax = axs[1]
    im1 = ax.imshow(activeP.T, aspect="auto", origin="lower",
                    extent=[tP[0], tP[-1], -0.5, N_P-0.5],
                    cmap="Greys", vmin=0, vmax=1,
                    interpolation="nearest")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Node index")
    ax.set_title("PCS constraint activation (ref)", fontsize=10)
    ax.set_yticks(range(N_P))
    cbar1 = fig.colorbar(im1, ax=ax, ticks=[0, 1])
    cbar1.ax.set_yticklabels(["Off", "On"])

    plt.tight_layout()
