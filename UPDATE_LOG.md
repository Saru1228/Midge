# Update Log

## 2024-XX-XX
- Added clear namespaces under `swarm/` (`data`, `preprocess`, `coarse`, `fields`, `observables`, `models`, `pde`) that re-export existing functions for a cleaner API while keeping old imports working.
- Deduplicated the KNN Gaussian Laplacian helper into `swarm/pde/common.py` and updated PDE fitting modules to reuse it.
- Refreshed `swarm/__init__.py` to expose the new namespaces, keep backward-compatible re-exports, and make `load_and_prepare` merge dict inputs before preprocessing.
