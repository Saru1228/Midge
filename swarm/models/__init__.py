"""
Modeling and inference utilities: MEM, RG scaling, and parameter fitting.
"""

from ..MEM import compute_C_data, compute_polarization, C_model, F
from ..RG import select_time_window, clean_vectors, knn_variance, power_law, power_law_fit
from ..LRG import analyze_metric_scaling
from ..rgfitting import (
    laplacian_vector_field,
    fit_D_gamma,
    gaussian_smooth_field,
    topo_laplacian_vector,
    fit_D_gamma_topological,
    scan_D_gamma_topological,
)

__all__ = [
    # MEM
    "compute_C_data",
    "compute_polarization",
    "C_model",
    "F",
    # RG scaling
    "select_time_window",
    "clean_vectors",
    "knn_variance",
    "power_law",
    "power_law_fit",
    "analyze_metric_scaling",
    # PDE/RG fitting
    "laplacian_vector_field",
    "fit_D_gamma",
    "gaussian_smooth_field",
    "topo_laplacian_vector",
    "fit_D_gamma_topological",
    "scan_D_gamma_topological",
]
