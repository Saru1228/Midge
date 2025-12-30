# ============================================================
# Multi-time L-PDE scan for v / a / j
#   Requires:
#     - fit_vel_pde_lagrangian
#     - fit_acc_pde_lagrangian
#     - fit_jerk_pde_lagrangian
#   to be already defined in the workspace.
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from swarm.lagranginpde import *


def scan_l_pde_over_time(
    df,
    id_col="id",
    pos_cols=("x","y","z"),
    v_cols=("vx","vy","vz"),
    a_cols=("ax","ay","az"),
    j_cols=("jx","jy","jz"),
    length_scale=15.0,
    k_neighbors=10,
    n_samples=20,
    verbose=True
):
    """
    在多个时间点 t0 上，分别拟合：
      - velocity L-PDE: a ≈ D_v Δv - γ_v v
      - acceleration L-PDE: da/dt ≈ D Δa - γ a
      - jerk L-PDE: dj/dt ≈ D_j Δj - γ_j j

    参数:
      - df: 包含 t, id, x,y,z, vx,vy,vz, ax,ay,az, jx,jy,jz 的 DataFrame
      - n_samples: 从时间轴上均匀抽样多少个 t0
      - length_scale, k_neighbors: KNN Laplacian 的参数

    返回:
      dict，包含每个 t0 的 ratio 和拟合参数
    """
    df = df.copy()
    t_vals = np.sort(df["t"].unique())
    if len(t_vals) < 3:
        raise RuntimeError("Not enough time frames for time scan.")

    # 选取 n_samples 个 t0 index（避开最后一帧，因为需要 t1）
    idx_candidates = np.arange(0, len(t_vals)-1)
    if n_samples >= len(idx_candidates):
        idx_list = idx_candidates
    else:
        idx_list = np.linspace(0, len(idx_candidates)-1, n_samples, dtype=int)
        idx_list = np.unique(idx_list)

    t0_list = []
    dt_list = []

    vel_ratio_list = []
    acc_ratio_list = []
    jerk_ratio_list = []

    vel_D_list, vel_gamma_list   = [], []
    acc_D_list, acc_gamma_list   = [], []
    jerk_D_list, jerk_gamma_list = [], []

    for idx0 in idx_list:
        t0 = t_vals[idx0]
        # 让各 fit_* 函数自己选 t1（默认下一帧），dt=None
        if verbose:
            print(f"\n===== L-PDE scan at t0 = {t0} (index {idx0}) =====")

        # velocity L-PDE
        try:
            res_v = fit_vel_pde_lagrangian(
                df,
                t0=t0,
                dt=None,
                id_col=id_col,
                pos_cols=pos_cols,
                v_cols=v_cols,
                a_cols=a_cols,
                length_scale=length_scale,
                k_neighbors=k_neighbors
            )
            vel_ratio = float(res_v["residual_ratio"])
            vel_D     = float(res_v["D_v"])
            vel_gamma = float(res_v["gamma_v"])
        except Exception as e:
            if verbose:
                print("  [vel-LPDE] failed:", e)
            vel_ratio, vel_D, vel_gamma = np.nan, np.nan, np.nan

        # acceleration L-PDE
        try:
            res_a = fit_acc_pde_lagrangian(
                df,
                t0=t0,
                dt=None,
                id_col=id_col,
                pos_cols=pos_cols,
                a_cols=a_cols,
                length_scale=length_scale,
                k_neighbors=k_neighbors
            )
            acc_ratio = float(res_a["residual_ratio"])
            acc_D     = float(res_a["D_hat"])
            acc_gamma = float(res_a["gamma_hat"])
        except Exception as e:
            if verbose:
                print("  [acc-LPDE] failed:", e)
            acc_ratio, acc_D, acc_gamma = np.nan, np.nan, np.nan

        # jerk L-PDE
        try:
            res_j = fit_jerk_pde_lagrangian(
                df,
                t0=t0,
                dt=None,
                id_col=id_col,
                pos_cols=pos_cols,
                j_cols=j_cols,
                length_scale=length_scale,
                k_neighbors=k_neighbors
            )
            jerk_ratio = float(res_j["residual_ratio"])
            jerk_D     = float(res_j["D_j"])
            jerk_gamma = float(res_j["gamma_j"])
        except Exception as e:
            if verbose:
                print("  [jerk-LPDE] failed:", e)
            jerk_ratio, jerk_D, jerk_gamma = np.nan, np.nan, np.nan

        # 统一时间和 dt（从其中一个结果取）
        dt_here = None
        for res in (locals().get("res_v"), locals().get("res_a"), locals().get("res_j")):
            if isinstance(res, dict) and "dt" in res:
                dt_here = float(res["dt"])
                break

        t0_list.append(float(t0))
        dt_list.append(dt_here)

        vel_ratio_list.append(vel_ratio)
        vel_D_list.append(vel_D)
        vel_gamma_list.append(vel_gamma)

        acc_ratio_list.append(acc_ratio)
        acc_D_list.append(acc_D)
        acc_gamma_list.append(acc_gamma)

        jerk_ratio_list.append(jerk_ratio)
        jerk_D_list.append(jerk_D)
        jerk_gamma_list.append(jerk_gamma)

    result = {
        "t0": np.array(t0_list),
        "dt": np.array(dt_list),

        "vel_ratio": np.array(vel_ratio_list),
        "vel_D": np.array(vel_D_list),
        "vel_gamma": np.array(vel_gamma_list),

        "acc_ratio": np.array(acc_ratio_list),
        "acc_D": np.array(acc_D_list),
        "acc_gamma": np.array(acc_gamma_list),

        "jerk_ratio": np.array(jerk_ratio_list),
        "jerk_D": np.array(jerk_D_list),
        "jerk_gamma": np.array(jerk_gamma_list),
    }

    return result


def plot_l_pde_ratios_over_time(scan_result):
    """
    根据 scan_l_pde_over_time 的结果，画出
    vel / acc / jerk 三条 residual ratio vs t0 的曲线。
    """
    t0 = scan_result["t0"]
    vel_r = scan_result["vel_ratio"]
    acc_r = scan_result["acc_ratio"]
    jerk_r = scan_result["jerk_ratio"]

    plt.figure(figsize=(7,5))
    plt.plot(t0, vel_r, "o-", label="vel-LPDE ratio (a ~ Δv,v)")
    plt.plot(t0, acc_r, "s-", label="acc-LPDE ratio (ȧ ~ Δa,a)")
    plt.plot(t0, jerk_r, "^-", label="jerk-LPDE ratio (ĵ ~ Δj,j)")
    plt.xlabel("t0")
    plt.ylabel("Residual ratio")
    plt.title("Lagrangian PDE closure vs time")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 简单打印平均值，方便快速解读
    print("\n=== Mean residual ratios over time ===")
    print("  mean vel-LPDE  ratio =", np.nanmean(vel_r))
    print("  mean acc-LPDE  ratio =", np.nanmean(acc_r))
    print("  mean jerk-LPDE ratio =", np.nanmean(jerk_r))
