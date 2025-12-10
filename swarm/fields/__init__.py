"""
Field operations on coarse-grained grids: derivatives, spectra, and decompositions.
"""

from ..hydrodynamic_derivatives import (
    get_spacings,
    gradient_scalar,
    divergence,
    curl,
    laplacian_scalar,
    gradient_energy,
    compute_velocity_correlation,
    compute_structure_factor_3D,
    compute_S_v_qw,
    laplacian_field,
)
from ..Helmholtz import helmholtz_decompose_fft

__all__ = [
    "get_spacings",
    "gradient_scalar",
    "divergence",
    "curl",
    "laplacian_scalar",
    "gradient_energy",
    "compute_velocity_correlation",
    "compute_structure_factor_3D",
    "compute_S_v_qw",
    "laplacian_field",
    "helmholtz_decompose_fft",
]
