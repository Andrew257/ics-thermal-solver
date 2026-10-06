# examples/run_benchmark_tables.py
"""
Benchmark runner producing the performance tables of Section 7.

Saves CSV files directly to the project root, named to match the
table numbers in the manuscript:
  - table_02_testB_performance.csv   (Table 2, Section 7.2)
  - table_03_testC_performance.csv   (Table 3, Section 7.3)

Usage:
  python examples/run_benchmark_tables.py
"""

import sys
from pathlib import Path
import csv
import time
import numpy as np

# Ensure project root and src/ are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

for p in (REPO_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ics_thermal.config import run_test_case


HEADERS = ["N", "CPU_ICS [s]", "Newton iterations", "CPU_PCS [s]", "PCS steps", "Speedup (PCS/ICS)"]


def format_int(val):
    return f"{int(val):,}".replace(",", " ")


def save_csv(filename, headers, rows):
    out_path = REPO_ROOT / filename
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"[+] Exported CSV to: {out_path}")


def print_table(title, headers, rows):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(val)))

    header_line = " | ".join(
        f"{headers[i]:<{col_widths[i]}}" if i == 0 else f"{headers[i]:>{col_widths[i]}}"
        for i in range(len(headers))
    )
    sep_line = "-+-".join("-" * w for w in col_widths)

    print("\n" + "=" * len(header_line))
    print(f" {title}")
    print("=" * len(header_line))
    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(
            f"{str(row[i]):<{col_widths[i]}}" if i == 0 else f"{str(row[i]):>{col_widths[i]}}"
            for i in range(len(row))
        ))
    print("=" * len(header_line) + "\n")


def run_benchmark_series(test_name, N_list, t_final, dt_ics, h_pcs):
    rows = []
    print(f"\n>>> Running Benchmark Sweep: Test {test_name} "
          f"(t_final={t_final}s, dt_ics={dt_ics}s, h_pcs={h_pcs}s)")
    print("-" * 75)

    for N in N_list:
        print(f"[*] Solving for N = {N} nodes...")

        # 1. Run ICS
        t0 = time.perf_counter()
        ics = run_test_case(test_name, N=N, solver="ICS", t_final=t_final, dt_ics=dt_ics)
        cpu_ics = time.perf_counter() - t0
        newton_iters = int(np.sum(ics["n_iter"]))

        # 2. Run PCS
        t0 = time.perf_counter()
        pcs = run_test_case(test_name, N=N, solver="PCS", t_final=t_final, h_pcs=h_pcs)
        cpu_pcs = time.perf_counter() - t0

        # Step count: prefer 'icount' from pcs dict, fallback to length of 't'
        pcs_steps = pcs.get("icount", len(pcs["t"]))

        # 3. Compute speedup
        speedup = (cpu_pcs / cpu_ics) if cpu_ics > 0 else 0.0

        # Format steps string (e.g. 5x10^6 if huge)
        if pcs_steps >= 1_000_000:
            steps_str = (r"$5 \times 10^6$"
                         if abs(pcs_steps - 5_000_000) < 500_000
                         else format_int(pcs_steps))
        else:
            steps_str = format_int(pcs_steps)

        row = [
            N,
            f"{cpu_ics:.1f}",
            format_int(newton_iters),
            f"{cpu_pcs:.1f}" if cpu_pcs < 1000 else f"{cpu_pcs:,.1f}".replace(",", " "),
            steps_str,
            f"{speedup:.2f}" if speedup < 10.0 else f"{speedup:.1f}"
        ]
        rows.append(row)
        print(f"    -> N={N:3d} | ICS: {row[1]}s ({row[2]} iters) | "
              f"PCS: {row[3]}s ({row[4]} steps) | Speedup: {row[5]}x")

    return rows


def main():
    # Chain lengths tested in the manuscript
    N_SWEEP = [10, 20, 50, 100, 200]

    # --- 1. Table 2: Test B ------------------------------------------
    rows_t2 = run_benchmark_series(
        test_name="B",
        N_list=N_SWEEP,
        t_final=13.0,
        dt_ics=0.05,
        h_pcs=0.0002
    )
    print_table("Table 2: Test B Performance", HEADERS, rows_t2)
    save_csv("table_02_testB_performance.csv", HEADERS, rows_t2)

    # --- 2. Table 3: Test C ------------------------------------------
    # Note: for N = 100 and N = 200, PCS with h_pcs = 2.5e-6 executes
    # ~5 million steps and takes several minutes per run.
    rows_t3 = run_benchmark_series(
        test_name="C",
        N_list=N_SWEEP,
        t_final=13.0,
        dt_ics=0.05,
        h_pcs=2.5e-6
    )
    print_table("Table 3: Test C Performance", HEADERS, rows_t3)
    save_csv("table_03_testC_performance.csv", HEADERS, rows_t3)


if __name__ == "__main__":
    main()
