"""
Observable construction on particle or grid data: pair correlations, structure
factors, alignment metrics, and basic swarm statistics.
"""

from ..analysis import (
    analyze_local_alignment,
    compute_pair_correlation,
    compute_structure_factor,
    compute_velocity_structure_factor,
)
from ..stats import (
    track_lengths,
    gyration_radius,
    compute_relative_kinetic_energy,
    velocity_pca,
)

__all__ = [
    "analyze_local_alignment",
    "compute_pair_correlation",
    "compute_structure_factor",
    "compute_velocity_structure_factor",
    "track_lengths",
    "gyration_radius",
    "compute_relative_kinetic_energy",
    "velocity_pca",
]
