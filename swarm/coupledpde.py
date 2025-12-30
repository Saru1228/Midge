# ============================================================
# Coupled acc–jerk Lagrangian PDE fit
#
#  System:
#    (1)  ȧ = j
#    (2)  ĵ ≈ D ∇²a - γ a - η j
#
#  On particles (Lagrangian) using 3 consecutive frames t0<t1<t2.
# ============================================================

import numpy as np
import pandas as pd
from swarm.pde.common import knn_gaussian_laplacian


def fit_coupled_acc_jerk_pde_lagrangian(
    df,
    t1=None,
    id_col="id",
    pos_cols=("x","y","z"),
    a_cols=("ax","ay","az"),
    j_cols=("jx","jy","jz"),
    length_scale=15.0,
    k_neighbors=10
):
    """
    使用三帧 t0<t1<t2 的粒子轨迹拟合耦合系统：

        1)  ȧ(t1) ≈ j(t1)
        2)  ĵ(t1) ≈ D ∇² a(t1) - γ a(t1) - η j(t1)

    使用中心差分：
        ȧ(t1) ≈ (a(t2) - a(t0)) / (2 dt)
        ĵ(t1) ≈ (j(t2) - j(t0)) / (2 dt)

    返回:
      dict 含:
        - t0, t1, t2, dt
        - D_hat, gamma_hat, eta_hat
        - ratio_eq1  (ȧ vs j)
        - ratio_eq2  (Ĵ vs PDE)
    """
    df = df.copy()
    t_vals = np.sort(df["t"].unique())
    if len(t_vals) < 3:
        raise RuntimeError("Need at least 3 time frames for coupled PDE fit.")

    # -------- 选择中间帧 t1 和邻近 t0,t2 --------
    if t1 is None:
        idx1 = len(t_vals)//2
        t1   = t_vals[idx1]
    else:
        idx1 = int(np.argmin(np.abs(t_vals - t1)))
        t1   = t_vals[idx1]

    if idx1 == 0 or idx1 == len(t_vals)-1:
        raise RuntimeError("Chosen t1 has no both-sided neighbors; pick a middle frame.")

    t0 = t_vals[idx1 - 1]
    t2 = t_vals[idx1 + 1]
    dt = float(t2 - t1)   # assume uniform spacing, dt>0

    print(f"[Coupled acc–jerk L-PDE] t0={t0}, t1={t1}, t2={t2}, dt={dt}")

    df0 = df[df["t"] == t0].copy()
    df1 = df[df["t"] == t1].copy()
    df2 = df[df["t"] == t2].copy()

    # -------- 通过 ID 对齐三帧粒子 --------
    cols_keep = [id_col] + list(pos_cols) + list(a_cols) + list(j_cols)

    df01 = pd.merge(
        df0[cols_keep],
        df1[cols_keep],
        on=id_col,
        suffixes=("_0", "_1")
    )
    df012 = pd.merge(
        df01,
        df2[cols_keep],
        on=id_col
    )
    # 自动给第三帧列名加后缀 "_2"
    rename_map = {}
    for c in pos_cols + a_cols + j_cols:
        if c in df012.columns and not c.endswith(("_0","_1")):
            rename_map[c] = c + "_2"
    df012 = df012.rename(columns=rename_map)

    print("  matched particles (3 frames) =", len(df012))
    if len(df012) < 5:
        raise RuntimeError("Too few matched particles across 3 frames.")

    # positions at t1
    pos1 = df012[[c+"_1" for c in pos_cols]].values

    # a(t0), a(t1), a(t2)
    a0 = df012[[c+"_0" for c in a_cols]].values
    a1 = df012[[c+"_1" for c in a_cols]].values
    a2 = df012[[c+"_2" for c in a_cols]].values

    # j(t0), j(t1), j(t2)
    j0 = df012[[c+"_0" for c in j_cols]].values
    j1 = df012[[c+"_1" for c in j_cols]].values
    j2 = df012[[c+"_2" for c in j_cols]].values

    # -------- 时间导数 (中心差分) --------
    da_dt = (a2 - a0) / (2.0 * dt)
    dj_dt = (j2 - j0) / (2.0 * dt)

    # -------- Laplacian[a](t1) 在粒子位置 --------
    lap_a = knn_gaussian_laplacian(
        pos1,
        a1,
        length_scale=length_scale,
        k_neighbors=k_neighbors,
    )

    # -------- 清理 NaN --------
    mask = (
        np.isfinite(a1).all(axis=1) &
        np.isfinite(j1).all(axis=1) &
        np.isfinite(da_dt).all(axis=1) &
        np.isfinite(dj_dt).all(axis=1) &
        np.isfinite(lap_a).all(axis=1)
    )

    a1_use   = a1[mask]
    j1_use   = j1[mask]
    da_use   = da_dt[mask]
    dj_use   = dj_dt[mask]
    lap_use  = lap_a[mask]

    print("  used particles after masking:", len(a1_use))
    if len(a1_use) < 5:
        raise RuntimeError("Too few valid particles after masking.")

    # ===================================================
    # (1) Test ȧ ≈ j  这一条（定义关系的检查）
    # ===================================================
    da_mag  = np.linalg.norm(da_use, axis=1)
    res1    = da_use - j1_use
    res1_mag = np.linalg.norm(res1, axis=1)

    var_da  = np.nanmean(da_mag**2)
    var_res1 = np.nanmean(res1_mag**2)
    ratio_eq1 = var_res1 / var_da

    print(f"  [Eq.1  ȧ ≈ j]   ratio_eq1 = {ratio_eq1:.4f}")

    # ===================================================
    # (2) Fit ĵ ≈ D ∇²a - γ a - η j
    #      i.e.  dj_dt ≈ c1*lap_a + c2*a1 + c3*j1
    # ===================================================
    X = np.stack([
        lap_use.reshape(-1),
        a1_use.reshape(-1),
        j1_use.reshape(-1)
    ], axis=1)
    y = dj_use.reshape(-1)

    params, residuals, rank, svals = np.linalg.lstsq(X, y, rcond=None)
    c1, c2, c3 = params

    D_hat     = c1
    gamma_hat = -c2
    eta_hat   = -c3

    # 预测 ĵ 并算残差
    dj_pred = D_hat * lap_a + (-gamma_hat) * a1 + (-eta_hat) * j1
    dj_mag  = np.linalg.norm(dj_dt[mask], axis=1)
    res2    = dj_dt[mask] - dj_pred[mask]
    res2_mag = np.linalg.norm(res2, axis=1)

    var_dj  = np.nanmean(dj_mag**2)
    var_res2 = np.nanmean(res2_mag**2)
    ratio_eq2 = var_res2 / var_dj

    print(f"  [Eq.2  ĵ PDE]  D={D_hat:.3f}, γ={gamma_hat:.3f}, η={eta_hat:.3f}, ratio_eq2={ratio_eq2:.4f}")

    return {
        "t0": float(t0),
        "t1": float(t1),
        "t2": float(t2),
        "dt": float(dt),
        "D_hat": float(D_hat),
        "gamma_hat": float(gamma_hat),
        "eta_hat": float(eta_hat),
        "ratio_eq1": float(ratio_eq1),
        "ratio_eq2": float(ratio_eq2),
        "n_particles": int(len(a1_use))
    }
