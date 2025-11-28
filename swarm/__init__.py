"""
swarm: Core library for insect swarm kinematics, statistics, and field-theory preparation.

This package provides:
- Data loading utilities (read swarm datasets)
- Kinematic preprocessing (velocity, acceleration, jerk)
- Basic statistical physics measurements
- Alignment and swarm-structure analysis
- Smooth trajectory tools

Designed as the foundation of:
- Coarse-grained hydrodynamic fields
- Toner–Tu parameter inference
- RG scaling analysis
- Reynolds jerk-driven swarm dynamics
- Agent-based model validation

Author: Yp_Hou
Version: 0.1.0
"""

__version__ = "0.1.0"

# ---- Re-export key functions for a clean public API ---- #

# Data IO
from .io import (
    read_swarm_batch,
    _read_one_file,
)

# Kinematics & preprocessing
from .kinematics import (
    center_by_com_each_frame,
    add_speed_accel,
    add_jerk,
    preprocess_full,
)

# Statistics
from .stats import (
    track_lengths,
    gyration_radius,
    compute_relative_kinetic_energy,
    velocity_pca,
)

# Higher-level analysis
from .analysis import (
    analyze_local_alignment,
)

# Utils
from .utils import (
    smooth_signal,
    apply_smoothing,
    slice_time_window,
    merge_dict_of_dfs,
)


# ---- Convenience: unified top-level loader-preprocessor ---- #

def load_and_prepare(folder, start=1, end=19, prefix="Ob", ext=".txt"):
    """
    High-level interface:
    Load a swarm dataset and compute:
    - centered trajectories
    - velocity & acceleration
    - jerk

    Equivalent to:
        df = read_swarm_batch(...)
        df = preprocess_full(df)

    Example:
        import swarm
        df = swarm.load_and_prepare("dataset_folder")
    """
    df = read_swarm_batch(folder, start=start, end=end, prefix=prefix, ext=ext)
    return preprocess_full(df)
