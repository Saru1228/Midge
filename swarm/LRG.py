# ============================================================
# Metric (L-grid) variance scaling for v / a / j
# Works with scalar magnitudes, so uses clean_scalar()
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from swarm.RG import *


def analyze_metric_scaling(df_win, L_list,
                           pos_cols=('x','y'),
                           v_cols=('vx','vy','vz'),
                           a_cols=('ax','ay','az'),
                           j_cols=('jx','jy','jz')):

    df = df_win.copy()
    L_arr = np.array(L_list, dtype=float)

    # ---------- 工具：用于 scalar magnitude ----------
    def clean_scalar(pos, mag):
        mask = ~np.isnan(mag)
        return pos[mask], mag[mask]

    # ---------- 提取坐标 ----------
    pos = df[list(pos_cols)].values

    vel_vec  = df[list(v_cols)].values
    acc_vec  = df[list(a_cols)].values
    jerk_vec = df[list(j_cols)].values

    # magnitude
    vmag = np.linalg.norm(vel_vec, axis=1)
    amag = np.linalg.norm(acc_vec, axis=1)
    jmag = np.linalg.norm(jerk_vec, axis=1)

    # 清洗 scalar
    pos_v, vmag = clean_scalar(pos, vmag)
    pos_a, amag = clean_scalar(pos, amag)
    pos_j, jmag = clean_scalar(pos, jmag)

    print(f"[Metric] N_v={len(pos_v)}, N_a={len(pos_a)}, N_j={len(pos_j)}")

    # ---------- metric coarse-graining ----------
    def var_L(pos_xy, mag, L):
        x = pos_xy[:,0]
        y = pos_xy[:,1]
        x_edges = np.arange(x.min(), x.max()+L, L)
        y_edges = np.arange(y.min(), y.max()+L, L)

        counts, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
        sums, _, _   = np.histogram2d(x, y, bins=(x_edges, y_edges),
                                      weights=mag)

        mask = counts > 0
        if not np.any(mask):
            return np.nan

        mean_mag = sums[mask] / counts[mask]
        return np.mean(mean_mag**2)

    # ---------- compute scaling ----------
    v2_list = [var_L(pos_v, vmag, L) for L in L_arr]
    a2_list = [var_L(pos_a, amag, L) for L in L_arr]
    j2_list = [var_L(pos_j, jmag, L) for L in L_arr]

    # ---------- fit scaling exponent ----------
    alpha_v, params_v = power_law_fit(L_arr, np.array(v2_list))
    alpha_a, params_a = power_law_fit(L_arr, np.array(a2_list))
    alpha_j, params_j = power_law_fit(L_arr, np.array(j2_list))

    print(f"[Metric] alpha_v = {alpha_v:.3f}")
    print(f"[Metric] alpha_a = {alpha_a:.3f}")
    print(f"[Metric] alpha_j = {alpha_j:.3f}")

    # ---------- plot ----------
    def _plot_metric(L_arr, data, params, title, ylabel):
        plt.figure(figsize=(6,5))
        plt.loglog(L_arr, data, 'o', label='data')
        plt.loglog(L_arr, power_law(L_arr, *params),
                   '-', label=f'fit α={params[1]:.3f}')
        plt.xlabel("L (metric scale)")
        plt.ylabel(ylabel)
        plt.grid(True, which="both")
        plt.title(title)
        plt.legend()
        plt.show()

    _plot_metric(L_arr, v2_list, params_v,
                 "Velocity variance scaling (Metric)", "<v^2>(L)")
    _plot_metric(L_arr, a2_list, params_a,
                 "Acceleration variance scaling (Metric)", "<a^2>(L)")
    _plot_metric(L_arr, j2_list, params_j,
                 "Jerk variance scaling (Metric)", "<j^2>(L)")

    return {"v": alpha_v, "a": alpha_a, "j": alpha_j}
