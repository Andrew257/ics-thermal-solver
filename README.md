# ICS Thermal Solver

**Constraint-Aware Execution Semantics for Nonlinear Thermal Models**  
*Frozen release: v1.0*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: >=3.9](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://www.python.org/)

This repository contains the reference implementation of the **Implicit Constraint Solver (ICS)** and the reference **Projected Constraint Solver (PCS)** described in the manuscript:

> **"Constraint-Aware Execution Semantics for Nonlinear Thermal Models"**  
> Andrew Parry, Matthieu Simon, Nicolas Maquignon, Giovanni Sosio (*International Journal of Modelling and Simulation*, 2026).

The tag `v1.0` archives the exact codebase used to generate all figures, benchmarks, and comparative performance data reported in the paper.

---

## Overview

Mainstream physical modeling and continuous simulation environments (such as Modelica and Simulink) typically enforce unilateral state limits via zero-crossing event localization followed by state projection. This approach can cause solver-dependent event chatter and discrete energy imbalances. 

The ICS framework provides:
- **Embedded Unilateral Constraints:** Active-set Newton iterations integrated directly into a backward Euler formulation.
- **Physical Multipliers:** Algebraic constraint heat flows ($Q_i^{\mathrm{c}}$) entering the discrete energy balance directly, preserving exact thermodynamic invariants.
- **Coupled Nonlinear Transport:** Simultaneous handling of temperature-dependent heat capacities and quartic radiation coupling ($\propto T^4$).
- **Stiff Regime Robustness:** Stable, deterministic macrostepping without heuristic state resets or event-search stalls.

---

## Repository Structure

```text
ics-thermal-solver/
│
├── src/
│   └── ics_thermal/               # Core solver and modeling modules
│       ├── active_set.py          # Active-set state detection and projection logic
│       ├── config.py              # Test cases (A, B, C) setup and driver dispatcher
│       ├── graph.py               # Thermal network graph topology definitions
│       ├── ics_driver.py          # Implicit Constraint Solver (ICS) driver
│       ├── jacobian.py            # Sparse Jacobian assembly & structure reuse
│       ├── newton.py              # Damped Newton-Raphson iteration loop
│       ├── pcs_driver.py          # Projected Constraint Solver (PCS) reference driver
│       ├── physics.py             # Conduction, radiation, and capacity laws
│       └── residual.py            # Discrete energy balance residuals
│
├── tests/                         # Unit tests and component verification
│   ├── test_active_set.py         # Unit tests for active-set switching
│   ├── test_config.py             # Verification of test parameter sets
│   └── test_ics_pcs_consistency.py# Numerical consistency checks between ICS and PCS
│
├── examples/                      # Reproduction and benchmark scripts
│   ├── plot_results.py            # Plotting routines (B&W/color publication-safe)
│   ├── run_tests.py               # Generates paper figures for Tests A, B, and C
│   └── run_benchmark_tables.py    # Generates Table 2 and Table 3 performance sweeps
│
├── table_2_test_b.csv             # Output benchmark data for Test B
├── table_3_test_c.csv             # Output benchmark data for Test C
├── requirements.txt               # Minimal Python dependencies
├── pyproject.toml                 # Package configuration
├── LICENSE                        # MIT License
└── README.md                      # Repository documentation
```

---

## Installation

### Prerequisites
- Python $\ge$ 3.9
- Git

### Setup
Clone the repository and install the dependencies:

```bash
git clone https://github.com/Andrew257/ics-thermal-solver.git
cd ics-thermal-solver
pip install -r requirements.txt
```

To install the package in editable mode:
```bash
pip install -e .
```

---

## Unit Testing

Run the test suite using `pytest`:

```bash
pytest
```

Or run individual test modules directly:

```bash
# Verify active-set switching mechanics
python tests/test_active_set.py

# Verify benchmark model configurations
python tests/test_config.py

# Verify numerical trajectory consistency between ICS and PCS
python tests/test_ics_pcs_consistency.py
```

---

## Reproducing Paper Results

All figures and performance tables from the paper can be reproduced using the scripts located in `examples/`.

### 1. Reproducing Figures (`examples/run_tests.py`)

The script `run_tests.py` runs the three canonical benchmark cases and renders all corresponding figures:
- **Test A ($N = 2$):** Baseline verification against continuous reference (Figures 2–7: temperatures, heat flows, constraint flows, phase-plane orbits, mode timelines, and Newton iterations).
- **Test B ($N = 10$):** Uniform chain with distributed, phase-shifted heating cycles (Figures 8–10: temperature propagation, activation raster, iteration profile).
- **Test C ($N = 10$):** Stiff network with quartic radiation and tight temperature bounds (Figures 11–14).

Run:
```bash
python examples/run_tests.py
```

*Note: Visualizations are generated using `examples/plot_results.py`, which utilizes high-contrast line styles and grayscale-safe patterns suitable for both online display and monochrome print.*

---

### 2. Reproducing Performance Tables (`examples/run_benchmark_tables.py`)

The script `run_benchmark_tables.py` performs the scalability sweeps across chain dimensions $N \in \{10, 20, 50, 100, 200\}$:
- **Table 2 (Test B):** Evaluates runtime scaling, Newton iterations, and reference step counts under asynchronous heating cycles.
- **Table 3 (Test C):** Measures execution times and order-of-magnitude speedup under severe numerical stiffness.

Run:
```bash
python examples/run_benchmark_tables.py
```

This will print the formatted ASCII tables in your console and export the results to:
- `table_2_test_b.csv`
- `table_3_test_c.csv`

---

## Versioning & Archival

The frozen version corresponding to the published paper is tagged as `v1.0`. To inspect or check out this exact release:

```bash
git fetch --tags
git checkout v1.0
```

---

## Citation

If you use this code or solver methodology in your research, please cite:

```bibtex
@article{parry2026constraint,
  title     = {Constraint-Aware Execution Semantics for Nonlinear Thermal Models},
  author    = {Parry, Andrew and Simon, Matthieu and Maquignon, Nicolas and Sosio, Giovanni},
  journal   = {International Journal of Modelling and Simulation},
  year      = {2026},
  publisher = {Taylor \& Francis}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
