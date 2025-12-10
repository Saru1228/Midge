"""
swarm: Core library for insect swarm kinematics, statistics, and field-theory preparation.

Reorganized namespaces (all still available from the top level):
  - data: raw I/O, smoothing, time slicing, merging
  - preprocess: centering + velocity/acc/jerk derivations
  - coarse: grid builders and Gaussian coarse-graining
  - fields: derivative operators and spectra on grids
  - observables: correlations, structure factors, and basic stats
  - models: MEM and RG-related fittings
  - pde: Lagrangian PDE fitting utilities

Author: Yp_Hou
Version: 0.1.0
"""

__version__ = "0.1.0"

# Namespaced subpackages
from . import data, preprocess, coarse, fields, observables, models, pde

# Backwards-compatible re-exports (kept flat for notebooks already written)
from .io import read_swarm_batch, _read_one_file
from .kinematics import center_by_com_each_frame, add_speed_accel, add_jerk, preprocess_full
from .stats import track_lengths, gyration_radius, compute_relative_kinetic_energy, velocity_pca
from .analysis import analyze_local_alignment, compute_pair_correlation, compute_structure_factor, compute_velocity_structure_factor
from .utils import smooth_signal, apply_smoothing, slice_time_window, merge_dict_of_dfs


def load_and_prepare(folder, start=1, end=19, prefix="Ob", ext=".txt"):
    """
    High-level interface:
    Load a swarm dataset and compute centered trajectories with
    velocity/acceleration/jerk columns.
    """
    dfs = read_swarm_batch(folder, start=start, end=end, prefix=prefix, ext=ext)
    if isinstance(dfs, dict):
        dfs = merge_dict_of_dfs(dfs)
    return preprocess_full(dfs)


__all__ = [
    # subpackages
    "data",
    "preprocess",
    "coarse",
    "fields",
    "observables",
    "models",
    "pde",
    # loaders / utils
    "read_swarm_batch",
    "_read_one_file",
    "smooth_signal",
    "apply_smoothing",
    "slice_time_window",
    "merge_dict_of_dfs",
    # preprocess
    "center_by_com_each_frame",
    "add_speed_accel",
    "add_jerk",
    "preprocess_full",
    "load_and_prepare",
    # stats / observables
    "track_lengths",
    "gyration_radius",
    "compute_relative_kinetic_energy",
    "velocity_pca",
    "analyze_local_alignment",
    "compute_pair_correlation",
    "compute_structure_factor",
    "compute_velocity_structure_factor",
]
