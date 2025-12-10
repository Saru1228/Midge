# ============================================================
# Jerk-driven Scaling (Success Version)
# Time-window selection + KNN coarse-graining + power-law fit
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import curve_fit


# ============================================================
# 0. 工具函数（通用，不入库，可随时迁移）
# ============================================================

def select_time_window(df, dt=0.5):
    """选择中间时刻 ± dt 的时间窗口"""
    t_unique = np.sort(df['t'].unique())
    t0 = t_unique[len(t_unique)//2]
    df_win = df[(df['t'] >= t0 - dt) & (df['t'] <= t0 + dt)].reset_index(drop=True)
    print(f"[Time window] {t0-dt:.3f} ~ {t0+dt:.3f},  N={len(df_win)}")
    return df_win


def clean_vectors(pos, vec):
    """移除向量中的 NaN 行"""
    mask = ~np.isnan(vec).any(axis=1)
    return pos[mask], vec[mask]


def knn_variance(pos, vec, k):
    """给定 KNN 尺度 k，计算 coarse-grained variance <|vec_k|^2>"""
    nbrs = NearestNeighbors(n_neighbors=k).fit(pos)
    _, idx = nbrs.kneighbors(pos)
    vec_k = vec[idx].mean(axis=1)
    return np.mean(np.linalg.norm(vec_k, axis=1)**2)


def power_law(x, C, alpha):
    """幂律模型 C * x^(-alpha)"""
    return C * x**(-alpha)


def power_law_fit(k_arr, y_arr):
    """对 y(k) 做幂律拟合，返回 alpha"""
    popt, _ = curve_fit(power_law, k_arr.astype(float), y_arr)
    return popt[1], popt   # 返回 alpha 和全部参数
