"""
ICS / PCS thermal chain solvers.

This package implements:
- Implicit Constraint Solver (ICS) on general graphs
- Projected Constraint Solver (PCS) as explicit baseline
"""

from .graph import ICSGraph, build_chain_graph
from .physics import PhysicsAssembly
from .jacobian import JacobianStructure
from .active_set import update_active_set, apply_temperature_constraints_sparse
from .residual import assemble_residual_and_jacobian_sparse
from .newton import newton_step
from .ics_driver import run_ics, run_ics_chain
from .pcs_driver import run_pcs, run_pcs_chain
from .config import configure_test_case, run_test_case

__all__ = [
    "ICSGraph",
    "build_chain_graph",
    "PhysicsAssembly",
    "JacobianStructure",
    "update_active_set",
    "apply_temperature_constraints_sparse",
    "assemble_residual_and_jacobian_sparse",
    "newton_step",
    "run_ics",
    "run_ics_chain",
    "run_pcs",
    "run_pcs_chain",
    "configure_test_case",
    "run_test_case",
]
