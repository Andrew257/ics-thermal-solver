# examples/run_projection_study.py
"""
Demonstrates the projection defect claimed in the manuscript:
  (i)  zero-crossing event localisation is step-size dependent
  (ii) projection leaves the algebraic constraint flow inconsistent
       with the modified state, so the discrete energy balance fails

Runs a step-size sweep on the PCS driver and compares the energy
residual against the ICS driver at the manuscript macrostep.
Parameters come from ics_thermal.config so the accountant evaluates
the same equations the drivers solve.

Outputs (to the repository root), named to match the manuscript:
  table_04_newton_stats.csv        Table 4, Section 7.4
  table_05_energy_residual.csv     Table 5, Section 7.5
  table_06_event_timing.csv        supporting data, not in the paper
  fig15_energy_residual.png        Figure 15
  fig16_stepsize_convergence.png   Figure 16

Usage:
  python examples/run_projection_study.py
"""

import sys
from pathlib import Path
import csv
import time

import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# Path resolution
# ----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for _p in (REPO_ROOT, SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ics_thermal.config import run_test_case, configure_test_case
from ics_thermal.graph import build_chain_graph


# ================================================================
# 0. Test configuration
# ================================================================
# Benchmark step sizes matching Table 1 of the manuscript.
BENCHMARK_HPCS = {
    "B": 2e-4,      # 0.0002 s
    "C": 2.5e-6,    # 2.5e-6 s
}

# Step-size refinement lists used for the convergence study.
SWEEP_H = {
    "B": [4e-4, 2e-4, 1e-4],
    "C": [1e-5, 5e-6, 2.5e-6],
}

DT_ICS  = 0.05
T_FINAL = 13.0

# Start-up transient: run_pcs records no constraint flow on its first
# step and uses the unclipped state as its reference on the following
# steps. For the stiff case that transient extends over several hundred
# intervals, so a fixed offset is applied before plotting.
STARTUP_SKIP = {"B": 0, "C": 200}


def fmt_h(h):
    """Compact, non-rounding formatting of a step size for labels."""
    if h >= 1e-4:
        return f"{h:g}"
    mant, exp = f"{h:.1e}".split("e")
    return f"{mant}e{int(exp)}"


# ================================================================
# 1. Energy residual accountant
# ================================================================
def energy_residual(result, params, h, driver="pcs"):
    """
    Discrete energy balance residual per node, per step:

        dE_i^k = C_i(T_i^{k+1}) * (T_i^{k+1} - T_i^k)
                 - h * ( -sum_j Q_ij^{k+1} + Qc_i^{k+1} + S_i )

    `params` must be the SAME tuple the driver was run with.

    driver="ics": ics_driver records the POST-step state at index k.
    driver="pcs": run_pcs records the PRE-step state at index k.
    """
    C0, C1, k1, k2, T_low, T_high, amp, period, phase = params

    t  = result["t"]
    T  = result["T"]
    Q  = result["Q"]
    Qc = result["Qc"]

    n = len(t)
    N = T.shape[1]
    graph = build_chain_graph(N)

    def C(i, Tval):
        return C0[i] + C1[i] * Tval

    def S(i, tval):
        if period[i] == 0.0:
            return 0.0
        return amp[i] * np.sin(2.0 * np.pi * tval / period[i] + phase[i])

    resid = np.zeros((n - 1, N))
    for k in range(n - 1):
        t_src = t[k + 1] if driver == "ics" else t[k]
        for i in range(N):
            sumQ_new = 0.0
            for e, sign in graph.node_edges[i]:
                sumQ_new += sign * Q[k + 1, e]

            lhs = C(i, T[k + 1, i]) * (T[k + 1, i] - T[k, i])
            rhs = h * (-sumQ_new + Qc[k + 1, i] + S(i, t_src))
            resid[k, i] = lhs - rhs

    return resid


def clip_events(result, T_low, T_high, tol=1e-9):
    """Boolean mask of steps where a node sits exactly on a bound."""
    T = result["T"]
    lo = np.asarray(T_low)[None, :]
    hi = np.asarray(T_high)[None, :]
    return (np.abs(T - lo) < tol) | (np.abs(T - hi) < tol)


def first_activation_time(result, node):
    mode = result["mode"]
    bit = 1 << node
    hits = np.where((mode & bit) != 0)[0]
    return float(result["t"][hits[0]]) if len(hits) else None


# ================================================================
# 2. Newton diagnostics
# ================================================================
def collect_newton_stats(ics_res):
    """Extract the Newton counters from an ICS result dict."""
    st = ics_res.get("newton_stats")
    if not st:
        return {
            "iterations": "", "steps_taken": "", "full_steps": "",
            "backtracked": "", "releases": "", "full_step_fraction": "",
        }
    steps = st["full_steps"] + st["damped_steps"]
    pct_full = (100.0 * st["full_steps"] / steps) if steps else 0.0
    return {
        "iterations":          st["iterations"],
        "steps_taken":         steps,
        "full_steps":          st["full_steps"],
        "backtracked":         st["damped_steps"],
        "releases":            st["releases"],
        "full_step_fraction":  round(pct_full, 4),
    }


def print_newton_stats(stats):
    if stats["steps_taken"] == "":
        print("    ICS Newton: no statistics recorded")
        return
    print(f"    ICS Newton: {stats['iterations']} iterations, "
          f"{stats['steps_taken']} steps taken, "
          f"full-step fraction = {stats['full_step_fraction']:.2f}%, "
          f"backtracked = {stats['backtracked']}, "
          f"releases = {stats['releases']}")


# ================================================================
# 3. Sweeps
# ================================================================
def run_stepsize_sweep(test_name, N, h_list, t_final=T_FINAL):
    params = configure_test_case(test_name, N)
    rows, trajectories = [], {}

    for h in h_list:
        t0 = time.perf_counter()
        res = run_test_case(test_name, N=N, solver="PCS",
                            t_final=t_final, h_pcs=h)
        wall = time.perf_counter() - t0

        dE = energy_residual(res, params, h, driver="pcs")
        clips = clip_events(res, params[4], params[5])

        k0 = STARTUP_SKIP.get(test_name, 0)
        k0 = min(k0, len(dE) - 1)
        dE_ste = dE[k0:]

        max_abs = float(np.max(np.abs(dE_ste))) if dE_ste.size else 0.0
        n_steps = dE.shape[0]
        n_clip = int(np.sum(np.any(clips[:-1], axis=1)))

        rows.append({
            "test": test_name, "N": N, "h": h,
            "steps": n_steps, "steps_analysed": dE_ste.shape[0],
            "clip_steps": n_clip,
            "clip_fraction": (n_clip / n_steps) if n_steps else 0.0,
            "max_abs_dE": max_abs,
            "median_abs_dE": float(np.median(np.abs(dE_ste))) if dE_ste.size else 0.0,
            "wall_s": wall,
        })
        trajectories[h] = res
        print(f"  [{test_name} N={N}] h={h:.2e}  steps={n_steps:>8d}  "
              f"clips={n_clip:>7d}  max|dE|={max_abs:.3e}  ({wall:.1f}s)")

    return rows, trajectories


def run_ics_reference(test_name, N, t_final=T_FINAL):
    params = configure_test_case(test_name, N)
    res = run_test_case(test_name, N=N, solver="ICS",
                        t_final=t_final, dt_ics=DT_ICS)
    dE = energy_residual(res, params, DT_ICS, driver="ics")

    mask = (res["mode"][:-1] == 0)
    if np.any(mask):
        bare = np.abs(dE[mask])
        print(f"    ICS unconstrained steps: "
              f"median={np.median(bare):.3e}  max={np.max(bare):.3e}")

    stats = collect_newton_stats(res)
    print_newton_stats(stats)

    return res, dE, stats


def run_timing_sweep(test_name, N, h_list, t_final=T_FINAL):
    nodes = sorted(set([0, N // 3, 2 * N // 3, N - 1])) if N >= 4 else list(range(N))
    timing = {i: {} for i in nodes}
    for h in h_list:
        res = run_test_case(test_name, N=N, solver="PCS",
                            t_final=t_final, h_pcs=h)
        for i in nodes:
            timing[i][h] = first_activation_time(res, i)
    return nodes, timing


# ================================================================
# 4. Output
# ================================================================
def save_csv(name, rows, fields):
    path = REPO_ROOT / name
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[+] {path}")


def plot_residual(ics_t, ics_dE, pcs_t, pcs_dE, test_name, N, h_pcs):
    """
    Figure 15 -- two-curve residual plot:
      - ICS at the manuscript macrostep
      - PCS at the exact benchmark step size for this test

    The start-up transient of the projected trace is excluded using
    STARTUP_SKIP, since run_pcs records no constraint flow on its first
    step and uses the unclipped state as its reference afterwards.
    """
    fig, ax = plt.subplots(figsize=(8.5, 3.9))

    ics_max = np.maximum(np.max(np.abs(ics_dE), axis=1), 1e-18)
    ax.semilogy(ics_t[:-1], ics_max,
                color="black", linestyle="-", linewidth=1.8,
                label=f"ICS (h_ICS = {DT_ICS:g} s)")

    a = np.abs(pcs_dE)
    k0 = min(STARTUP_SKIP.get(test_name, 0), len(a) - 1)
    pcs_max = np.maximum(np.max(a[k0:], axis=1), 1e-18)
    ax.semilogy(pcs_t[k0:-1], pcs_max,
                color="#555555", linestyle="--", linewidth=1.5,
                label="PCS (hpcs=" + fmt_h(h_pcs) + " s)")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Maximum energy residual $\max_i |\Delta E_i|$ [J]")
    ax.set_title(f"Discrete energy balance residual, Test {test_name} (N = {N})")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="center right", framealpha=0.95, fontsize=9)
    fig.tight_layout()

    out = REPO_ROOT / f"fig15_energy_residual_test{test_name}.png"
    fig.savefig(out, dpi=300)
    print(f"[+] {out}  (PCS trace starts at interval {k0})")
    plt.close(fig)


def plot_convergence(rows_by_test):
    """Figure 16 -- residual against step size, both tests."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    markers = {"B": "o", "C": "s"}

    for test_name, rows in rows_by_test.items():
        hs = [r["h"] for r in rows]
        ax.loglog(hs, [r["max_abs_dE"] for r in rows],
                  marker=markers.get(test_name, "o"), linestyle="-",
                  color="black" if test_name == "B" else "#555555",
                  label=f"Test {test_name} (max)")
        ax.loglog(hs, [r["median_abs_dE"] for r in rows],
                  marker=markers.get(test_name, "o"), linestyle="--",
                  color="black" if test_name == "B" else "#555555",
                  alpha=0.6, label=f"Test {test_name} (median)")

    ax.set_xlabel(r"Fine step size $h$ [s]")
    ax.set_ylabel(r"$|\Delta E|$  [J]")
    ax.set_title("Residual against step size\n"
                 "(a nonzero floor would indicate a persistent projection defect)")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out = REPO_ROOT / "fig16_stepsize_convergence.png"
    fig.savefig(out, dpi=300)
    print(f"[+] {out}")
    plt.close(fig)


def print_timing_table(nodes, timing, h_list, test_name, N):
    print(f"\nEvent timing, Test {test_name} (N = {N})")
    header = "node | " + " | ".join(f"h={fmt_h(h)}" for h in h_list) + " | drift"
    print("-" * len(header)); print(header); print("-" * len(header))
    for i in nodes:
        vals = [timing[i][h] for h in h_list]
        present = [v for v in vals if v is not None]
        drift = (max(present) - min(present)) if len(present) > 1 else float("nan")
        cells = " | ".join("  --  " if v is None else f"{v:6.3f}" for v in vals)
        print(f"{i:4d} | {cells} | {drift:.4f}")
    print()


# ================================================================
# 5. Main
# ================================================================
def main():
    print("=" * 72)
    print(" Projection study: energy balance and event localisation")
    print("=" * 72)

    all_rows, rows_by_test   = [], {}
    timing_rows, newton_rows = [], []

    for test_name, N in [("B", 10), ("C", 10)]:
        print(f"\n>>> Test {test_name} (N = {N})")
        h_list = SWEEP_H[test_name]

        # ---- ICS reference ---------------------------------------
        print("  ICS reference ...")
        ics_res, ics_dE, stats = run_ics_reference(test_name, N)
        ics_max = float(np.max(np.abs(ics_dE)))
        print(f"    ICS  max|dE| = {ics_max:.3e}")
        newton_rows.append({"test": test_name, "N": N, **stats})

        # ---- step-size sweep -------------------------------------
        rows, trajectories = run_stepsize_sweep(test_name, N, h_list)
        for r in rows:
            r["ics_max_abs_dE"] = ics_max
        rows_by_test[test_name] = rows
        all_rows.extend(rows)

        # ---- Figure 15 at the benchmark step size ----------------
        target_h = BENCHMARK_HPCS[test_name]
        if target_h in trajectories:
            res_target = trajectories[target_h]
        else:
            print(f"  PCS benchmark run at h = {fmt_h(target_h)} s ...")
            res_target = run_test_case(test_name, N=N, solver="PCS",
                                       t_final=T_FINAL, h_pcs=target_h)

        params = configure_test_case(test_name, N)
        dE_target = energy_residual(res_target, params, target_h, driver="pcs")
        plot_residual(ics_res["t"], ics_dE, res_target["t"],
                      dE_target, test_name, N, target_h)

        # ---- event timing ----------------------------------------
        nodes, timing = run_timing_sweep(test_name, N, h_list)
        print_timing_table(nodes, timing, h_list, test_name, N)

        for i in nodes:
            vals = [timing[i][h] for h in h_list]
            present = [v for v in vals if v is not None]
            row = {"test": test_name, "N": N, "node": i}
            for j in range(3):
                row[f"h_{j+1}"] = h_list[j] if j < len(h_list) else ""
                row[f"t_act_{j+1}"] = ("" if j >= len(vals) or vals[j] is None
                                       else vals[j])
            row["drift"] = ((max(present) - min(present))
                            if len(present) > 1 else "")
            timing_rows.append(row)

    # ---- persist --------------------------------------------------
    # Table 4 (Section 7.4)
    if newton_rows:
        save_csv("table_04_newton_stats.csv", newton_rows, [
            "test", "N", "iterations", "steps_taken",
            "full_steps", "backtracked", "releases", "full_step_fraction",
        ])

    # Table 5 (Section 7.5)
    save_csv("table_05_energy_residual.csv", all_rows, [
        "test", "N", "h", "steps", "steps_analysed", "clip_steps",
        "clip_fraction", "max_abs_dE", "median_abs_dE", "ics_max_abs_dE",
        "wall_s",
    ])

    # Supporting data, not appearing in the paper
    if timing_rows:
        save_csv("table_06_event_timing.csv", timing_rows, [
            "test", "N", "node", "h_1", "h_2", "h_3",
            "t_act_1", "t_act_2", "t_act_3", "drift",
        ])

    plot_convergence(rows_by_test)

    # ---- console summary ------------------------------------------
    print("\n" + "=" * 72)
    print(" Summary")
    print("=" * 72)
    for r in all_rows:
        print(f"  Test {r['test']}  h={r['h']:.1e}  "
              f"clips {r['clip_fraction']*100:6.2f}%  "
              f"max|dE|={r['max_abs_dE']:.3e}  "
              f"median={r['median_abs_dE']:.3e}  "
              f"ICS={r['ics_max_abs_dE']:.3e}")
    print()
    for n in newton_rows:
        print(f"  Newton {n['test']}: {n['full_step_fraction']}% full steps, "
              f"{n['releases']} releases")
    print()


if __name__ == "__main__":
    main()
