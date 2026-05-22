# examples/run_tests.py

import time
import matplotlib.pyplot as plt
from ics_thermal.config import run_test_case
from plot_results import (
    plot_temperatures,
    plot_heatflows,
    plot_modes,
    plot_newton_iterations,
    plot_compare_ics_pcs,
    plot_phase_planes_ics_pcs,
)

def run_test_A():
    print("\n=== Running Test A (N=2) ===")
    t0 = time.perf_counter()
    ics, pcs = run_test_case("A", N=2, solver="both",
                             t_final=13.0, dt_ics=0.05, h_pcs=0.0002)
    t1 = time.perf_counter()
    print(f"ICS CPU = {t1 - t0:.3f}s, Newton iters = {sum(ics['n_iter'])}")
    print(f"PCS steps = {pcs['icount']}")

    plot_compare_ics_pcs(ics, pcs, nodes=[0, 1])
    plt.show()

    plot_phase_planes_ics_pcs(ics, pcs)
    plt.show()

    plot_newton_iterations(ics)
    plt.show()

    return ics, pcs


def run_test_B():
    print("\n=== Running Test B (N=10) ===")
    t0 = time.perf_counter()
    ics = run_test_case("B", N=10, solver="ICS",
                        t_final=13.0, dt_ics=0.05)
    t1 = time.perf_counter()
    print(f"ICS CPU = {t1 - t0:.3f}s, Newton iters = {sum(ics['n_iter'])}")

    # PCS with coarse step for quick comparison
    t0 = time.perf_counter()
    pcs = run_test_case("B", N=10, solver="PCS",
                        t_final=13.0, h_pcs=0.0002)
    t1 = time.perf_counter()
    print(f"PCS CPU = {t1 - t0:.3f}s, steps = {pcs['icount']}")

    plot_compare_ics_pcs(ics, pcs, nodes=[0, 3, 6, 9])
    plt.show()
    plot_phase_planes_ics_pcs(ics, pcs)
    plt.show()
    plot_newton_iterations(ics)
    plt.show()
    return ics, pcs


def run_test_C():
    print("\n=== Running Test C (N=10) ===")
    t0 = time.perf_counter()
    ics = run_test_case("C", N=10, solver="ICS",
                        t_final=13.0, dt_ics=0.05)
    t1 = time.perf_counter()
    print(f"ICS CPU = {t1 - t0:.3f}s, Newton iters = {sum(ics['n_iter'])}")
    t0 = time.perf_counter()
    pcs = run_test_case("C", N=10, solver="PCS",
                        t_final=13.0, h_pcs=2.5e-6)
    t1 = time.perf_counter()
    print(f"PCS CPU = {t1 - t0:.3f}s, steps = {pcs['icount']}")

    plot_compare_ics_pcs(ics, pcs, nodes=[0, 3, 6, 9])
    plt.show()
    plot_phase_planes_ics_pcs(ics, pcs)
    plt.show()
    plot_newton_iterations(ics)
    plt.show()
    return ics, pcs


if __name__ == "__main__":
    run_test_A()
    run_test_B()
    run_test_C()
