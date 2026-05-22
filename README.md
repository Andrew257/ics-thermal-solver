ICS Thermal Solver
Constraint‑Aware Execution Semantics for Nonlinear Thermal Models  
Frozen version: v1.0

This repository contains the reference implementation of the implicit constraint solver (ICS) used in the JMSIC manuscript “Constraint‑Aware Execution Semantics for Nonlinear Thermal Models”.
The version tagged v1.0 corresponds exactly to the solver used to generate all results in the paper.

1. Overview
The solver implements:

nonlinear thermal dynamics

temperature‑dependent heat capacities

implicit constraint‑aware execution semantics

stable integration under stiff regimes

energy‑consistent updates

The implementation is lightweight, dependency‑minimal, and designed for reproducibility.

2. Repository Structure
Code
ics-thermal-solver/
│
├── src/                 # Core solver implementation
├── tests/               # Test A/B/C scripts
├── examples/            # Example models and reproduction scripts
├── README.md            # This file
├── requirements.txt     # Minimal dependencies
└── LICENSE              # MIT license
3. Installation
bash
pip install -r requirements.txt
Python ≥ 3.9 recommended.
4. Running the Solver
Test A — Active set
bash
python tests/test\_active\_set.py
Test B — Configuration
bash
python tests/test\_config.py
Test C — ICS and PCS consistency
bash
python tests/test\_ics\_pcs\_consistency.py
5. Reproducing Figures from the Paper
The figures are created using the script:

&#x20;   Run:

&#x20;   bash
python examples/run\_tests.py
All figures in the paper can be regenerated using this script.

6. Versioning
The solver version used in the manuscript is archived under:

Code
git tag v1.0
To check out the frozen version:

bash
git checkout v1.0
7. Citation
If you use this solver, please cite:

Code
Parry, A. (2026). Constraint‑Aware Execution Semantics for Nonlinear Thermal Models. JMSIC.
8. License
This project is released under the MIT License (see LICENSE file).

