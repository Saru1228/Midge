# ============================================================
# Lagrangian PDE fits for:
#   1) velocity-based PDE: a ≈ D_v Δv - γ_v v
#   2) jerk-based PDE: j_dot ≈ D_j Δj - γ_j j
#
#   Requirements:
#     df: must contain columns t, id, x,y,z, vx,vy,vz, ax,ay,az, jx,jy,jz
# ============================================================

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from swarm.pde.common import knn_gaussian_laplacian


# ============================================================
# 1) Velocity-based PDE:  a ≈ D_v Δv - γ_v v
# ============================================================
def fit_vel_pde_lagrangian(
    df,
    t0=None,
    dt=None,
    id_col="id",
    pos_cols=("x","y","z"),
    v_cols=("vx","vy","vz"),
    a_cols=("ax","ay","az"),
    length_scale=15.0,
    k_neighbors=10
):
    df = df.copy()
    t_vals = np.sort(df["t"].unique())

    # pick frames
    if t0 is None:
        idx0 = len(t_vals)//2
        t0 = t_vals[idx0]
    else:
        idx0 = int(np.argmin(np.abs(t_vals - t0)))
        t0 = t_vals[idx0]

    if dt is None:
        t1 = t_vals[idx0+1]
        dt = float(t1 - t0)
    else:
        t1 = t0 + dt
        idx1 = int(np.argmin(np.abs(t_vals - t1)))
        t1 = t_vals[idx1]
        dt = float(t1 - t0)

    df0 = df[df["t"] == t0].copy()
    df1 = df[df["t"] == t1].copy()

    # match IDs
    dfm = pd.merge(
        df0[[id_col]+list(pos_cols)+list(v_cols)+list(a_cols)],
        df1[[id_col]+list(pos_cols)+list(v_cols)+list(a_cols)],
        on=id_col,
        suffixes=("_0","_1")
    )
    print(f"[vel-LPDE] matched particles = {len(dfm)}")

    pos0 = dfm[[c+"_0" for c in pos_cols]].values
    v0   = dfm[[c+"_0" for c in v_cols]].values
    a0   = dfm[[c+"_0" for c in a_cols]].values

    # Laplacian of velocity
    lap_v = knn_gaussian_laplacian(
        pos0,
        v0,
        length_scale=length_scale,
        k_neighbors=k_neighbors,
    )

    # Fit: a ≈ c1 lap_v + c2 v
    mask = (
        np.isfinite(a0).all(axis=1) &
        np.isfinite(v0).all(axis=1) &
        np.isfinite(lap_v).all(axis=1)
    )
    a_use   = a0[mask]
    lap_use = lap_v[mask]
    v_use   = v0[mask]

    X = np.stack([lap_use.reshape(-1), v_use.reshape(-1)], axis=1)
    y = a_use.reshape(-1)

    params,_res,rank,_ = np.linalg.lstsq(X, y, rcond=None)
    c1, c2 = params

    D_v     = c1
    gamma_v = -c2

    # residual check
    a_pred = D_v * lap_v + (-gamma_v) * v0
    res    = a0 - a_pred

    a_mag  = np.linalg.norm(a0, axis=1)
    res_mag = np.linalg.norm(res, axis=1)

    var_a  = np.nanmean(a_mag[mask]**2)
    var_res = np.nanmean(res_mag[mask]**2)
    ratio = var_res / var_a

    print(f"  [vel-LPDE] D_v={D_v:.3f}, gamma_v={gamma_v:.3f}, ratio={ratio:.4f}")

    return {
        "t0":t0, "t1":t1, "dt":dt,
        "D_v":D_v, "gamma_v":gamma_v,
        "residual_ratio":ratio,
        "n_particles":len(a_use)
    }


# ============================================================
# 2) Jerk-based PDE:  j_dot ≈ D_j Δj - γ_j j
# ============================================================
def fit_jerk_pde_lagrangian(
    df,
    t0=None,
    dt=None,
    id_col="id",
    pos_cols=("x","y","z"),
    j_cols=("jx","jy","jz"),
    length_scale=15.0,
    k_neighbors=10
):
    df = df.copy()
    t_vals = np.sort(df["t"].unique())

    # select frames
    if t0 is None:
        idx0 = len(t_vals)//2
        t0 = t_vals[idx0]
    else:
        idx0 = int(np.argmin(np.abs(t_vals - t0)))
        t0 = t_vals[idx0]

    if dt is None:
        t1 = t_vals[idx0+1]
        dt = float(t1 - t0)
    else:
        t1 = t0 + dt
        idx1 = int(np.argmin(np.abs(t_vals - t1)))
        t1 = t_vals[idx1]
        dt = float(t1 - t0)

    df0 = df[df["t"] == t0].copy()
    df1 = df[df["t"] == t1].copy()

    # match particles
    dfm = pd.merge(
        df0[[id_col]+list(pos_cols)+list(j_cols)],
        df1[[id_col]+list(pos_cols)+list(j_cols)],
        on=id_col,
        suffixes=("_0","_1")
    )
    print(f"[jerk-LPDE] matched particles = {len(dfm)}")

    pos0 = dfm[[c+"_0" for c in pos_cols]].values
    j0   = dfm[[c+"_0" for c in j_cols]].values
    j1   = dfm[[c+"_1" for c in j_cols]].values

    # time derivative of jerk
    dj_dt = (j1 - j0) / dt

    # Laplacian of jerk field
    lap_j = knn_gaussian_laplacian(
        pos0,
        j0,
        length_scale=length_scale,
        k_neighbors=k_neighbors,
    )

    mask = (
        np.isfinite(j0).all(axis=1) &
        np.isfinite(dj_dt).all(axis=1) &
        np.isfinite(lap_j).all(axis=1)
    )
    j_use   = j0[mask]
    dj_use  = dj_dt[mask]
    lap_use = lap_j[mask]

    X = np.stack([lap_use.reshape(-1), j_use.reshape(-1)], axis=1)
    y = dj_use.reshape(-1)

    params,_res,rank,_ = np.linalg.lstsq(X, y, rcond=None)
    c1, c2 = params

    D_j     = c1
    gamma_j = -c2

    # residual check
    dj_pred = D_j * lap_j + (-gamma_j) * j0
    res     = dj_dt - dj_pred

    dj_mag  = np.linalg.norm(dj_dt, axis=1)
    res_mag = np.linalg.norm(res, axis=1)

    var_dj  = np.nanmean(dj_mag[mask]**2)
    var_res = np.nanmean(res_mag[mask]**2)
    ratio = var_res / var_dj

    print(f"  [jerk-LPDE] D_j={D_j:.3f}, gamma_j={gamma_j:.3f}, ratio={ratio:.4f}")

    return {
        "t0":t0, "t1":t1, "dt":dt,
        "D_j":D_j, "gamma_j":gamma_j,
        "residual_ratio":ratio,
        "n_particles":len(j_use)
    }

def fit_acc_pde_lagrangian(
    df,
    t0=None,
    dt=None,
    id_col="id",
    pos_cols=("x","y","z"),
    a_cols=("ax","ay","az"),
    length_scale=15.0,   # 类似你之前的 sigma，用作邻域长度尺度
    k_neighbors=10
):
    """
    Lagrangian-level fit of PDE:
        da/dt ≈ D * Laplacian[a] - γ * a
    using particle IDs.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: 't', id_col, pos_cols, a_cols.
    t0 : float or None
        Base time. If None, use middle frame.
    dt : float or None
        Time step. If None, infer from nearest next frame after t0.
    id_col : str
        Name of particle ID column.
    pos_cols : tuple
        Position columns, e.g. ('x','y','z').
    a_cols : tuple
        Acceleration columns, e.g. ('ax','ay','az').
    length_scale : float
        Length scale ℓ for Gaussian weights in Laplacian.
    k_neighbors : int
        Number of neighbors for KNN Laplacian.

    Returns
    -------
    result : dict
        {
          "t0": ...,
          "t1": ...,
          "dt": ...,
          "D_hat": ...,
          "gamma_hat": ...,
          "residual_ratio": ...,
          "n_particles": ...
        }
    """

    df = df.copy()
    t_vals = np.sort(df["t"].unique())

    # ----- choose t0, t1, dt -----
    if t0 is None:
        idx0 = len(t_vals)//2
        t0 = t_vals[idx0]
    else:
        # snap t0 to closest available frame
        idx0 = int(np.argmin(np.abs(t_vals - t0)))
        t0 = t_vals[idx0]

    if dt is None:
        if idx0+1 >= len(t_vals):
            raise ValueError("t0 is the last frame; choose earlier t0.")
        t1 = t_vals[idx0+1]
        dt = float(t1 - t0)
    else:
        # choose frame whose time is closest to t0+dt
        t_target = t0 + dt
        idx1 = int(np.argmin(np.abs(t_vals - t_target)))
        t1 = t_vals[idx1]
        dt = float(t1 - t0)

    print(f"[Lagrangian PDE fit] t0 = {t0}, t1 = {t1}, dt = {dt}")

    df0 = df[df["t"] == t0].copy()
    df1 = df[df["t"] == t1].copy()

    # ----- match particles by ID (inner join) -----
    df_merged = pd.merge(
        df0[[id_col] + list(pos_cols) + list(a_cols)],
        df1[[id_col] + list(pos_cols) + list(a_cols)],
        on=id_col,
        suffixes=("_0", "_1")
    )

    if len(df_merged) == 0:
        raise RuntimeError("No overlapping particle IDs between t0 and t1.")

    print("  matched particles =", len(df_merged))

    # positions at t0
    pos0 = df_merged[[c + "_0" for c in pos_cols]].values  # (N,3)
    # acceleration at t0, t1
    a0 = df_merged[[c + "_0" for c in a_cols]].values      # (N,3)
    a1 = df_merged[[c + "_1" for c in a_cols]].values      # (N,3)

    # time derivative da/dt
    da_dt = (a1 - a0) / dt   # (N,3)

    # ----- approximate Laplacian of a at particle positions -----
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors+1, len(pos0))).fit(pos0)
    distances, indices = nbrs.kneighbors(pos0)
    # indices: (N, k+1) including self at index 0

    N = len(pos0)
    lap_a = np.zeros_like(a0)

    for i in range(N):
        idx_nei = indices[i, 1:]         # exclude self
        r2 = distances[i, 1:]**2
        if len(idx_nei) == 0:
            continue

        # Gaussian weights
        w = np.exp(-r2 / (2.0 * length_scale**2))
        s = w.sum()
        if s <= 0:
            continue
        w /= s

        a_nei = a0[idx_nei]             # (k,3)
        # discrete Laplacian ~ Σ w (a_j - a_i) / ℓ^2
        lap_a[i] = (w[:, None] * (a_nei - a0[i])).sum(axis=0) / (length_scale**2)

    # ----- remove NaN / inf -----
    mask = (
        np.isfinite(a0).all(axis=1) &
        np.isfinite(da_dt).all(axis=1) &
        np.isfinite(lap_a).all(axis=1)
    )
    a0_use    = a0[mask]
    da_dt_use = da_dt[mask]
    lap_use   = lap_a[mask]

    print("  used particles after masking:", len(a0_use))

    if len(a0_use) < 5:
        raise RuntimeError("Too few valid particles after masking.")

    # ----- linear regression: da/dt ≈ c1 * lap_a + c2 * a -----
    X = np.stack([
        lap_use.reshape(-1),       # column 1: Laplacian[a]
        a0_use.reshape(-1)         # column 2: a
    ], axis=1)
    y = da_dt_use.reshape(-1)

    params, residuals, rank, svals = np.linalg.lstsq(X, y, rcond=None)
    c1, c2 = params

    D_hat     = c1
    gamma_hat = -c2

    # ----- residual check -----
    da_dt_pred = D_hat * lap_a + (-gamma_hat) * a0   # use full arrays

    da_mag   = np.linalg.norm(da_dt, axis=1)
    res_mag  = np.linalg.norm(da_dt - da_dt_pred, axis=1)

    var_da   = np.nanmean(da_mag[mask]**2)
    var_res  = np.nanmean(res_mag[mask]**2)
    ratio    = var_res / var_da

    print("\n[Fitted Lagrangian PDE parameters]")
    print(f"  D_hat     = {D_hat:.6f}")
    print(f"  gamma_hat = {gamma_hat:.6f}")
    print(f"  residual ratio (Lagrangian) = {ratio:.4f}")

    return {
        "t0": t0,
        "t1": t1,
        "dt": dt,
        "D_hat": D_hat,
        "gamma_hat": gamma_hat,
        "residual_ratio": ratio,
        "n_particles": len(a0_use)
    }

