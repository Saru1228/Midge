'''
alignment of group

'''
import numpy as np
import pandas as pd

def analyze_local_alignment(df, radius=0.02):
    """
    度量群体内部的一致性（alignment）。
    """
    ids = df["id"].unique()
    aligns = []

    for insect in ids:
        dfi = df[df["id"] == insect]
        others = df[df["id"] != insect]

        for t, row in dfi.iterrows():
            neigh = others[
                (others["t"] == row["t"]) &
                ((others["x"] - row["x"])**2 +
                 (others["y"] - row["y"])**2 +
                 (others["z"] - row["z"])**2 < radius**2)
            ]
            if len(neigh) > 0:
                v0 = np.array([row["vx"], row["vy"], row["vz"]])
                vs = neigh[["vx", "vy", "vz"]].values
                dot = vs @ v0 / (np.linalg.norm(v0) * np.linalg.norm(vs, axis=1))
                aligns.append(np.mean(dot))

    return np.mean(aligns) if len(aligns) > 0 else np.nan

def compute_pair_correlation(df, dr=5.0, r_max=None, min_particles=20):
    """
    计算 3D 昆虫群的 pair correlation function g(r)。

    参数：
        df : DataFrame
            需要包含列 ["x", "y", "z", "t"]，最好已做过中心化。
        dr : float
            径向 bin 宽度（与坐标单位一致；Sinhuber 数据一般是 mm）。
        r_max : float or None
            最大考虑的距离。如果为 None，则自动设为盒子最小边长的一半。
        min_particles : int
            跳过粒子数太少的帧（避免统计噪声）。

    返回：
        r_centers : 1D array
            每个 bin 的中心 r。
        g_r : 1D array
            对应的 g(r)。
    """
    # 只取需要的列，避免不必要的内存占用
    df_pos = df[["x", "y", "z", "t"]].dropna().copy()

    # 估计整体空间边界与平均体积/数密度
    x, y, z = df_pos["x"].values, df_pos["y"].values, df_pos["z"].values
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    zmin, zmax = z.min(), z.max()

    Lx, Ly, Lz = (xmax - xmin), (ymax - ymin), (zmax - zmin)
    V_box = Lx * Ly * Lz

    # 若未指定 r_max，则取最小边长的一半，避免边界效应过强
    if r_max is None:
        r_max = 0.5 * min(Lx, Ly, Lz)

    # 构建 r 的分箱
    bin_edges = np.arange(0.0, r_max + dr, dr)
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins = len(r_centers)

    # 初始化 histogram
    g_counts = np.zeros(n_bins, dtype=np.float64)

    # 按帧分组
    frames = np.sort(df_pos["t"].unique())
    n_frames_used = 0
    N_list = []
    V_list = []

    for t in frames:
        frame = df_pos[df_pos["t"] == t]
        coords = frame[["x", "y", "z"]].values
        N = coords.shape[0]
        if N < min_particles:
            continue  # 粒子太少的帧不计入统计

        # 这一帧的局部边界与体积（也可以用全局 V_box，这里给你一个可选思路）
        fxmin, fxmax = coords[:, 0].min(), coords[:, 0].max()
        fymin, fymax = coords[:, 1].min(), coords[:, 1].max()
        fzmin, fzmax = coords[:, 2].min(), coords[:, 2].max()
        fLx, fLy, fLz = (fxmax - fxmin), (fymax - fymin), (fzmax - fzmin)
        fV = max(fLx, 1e-9) * max(fLy, 1e-9) * max(fLz, 1e-9)

        # 计算该帧内所有粒子对的距离（上三角）
        # N 一般 O(10^2)，用广播即可
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=-1))
        iu = np.triu_indices(N, k=1)
        dists = dist_mat[iu]  # 只取 i<j 的距离

        # 对这帧的 pair 距离做 histogram
        hist, _ = np.histogram(dists, bins=bin_edges)
        g_counts += hist

        n_frames_used += 1
        N_list.append(N)
        V_list.append(fV)

    if n_frames_used == 0:
        raise RuntimeError("没有足够粒子的帧用于统计 g(r)（请检查 min_particles 或数据）。")

    # 统计平均粒子数与平均体积 → 平均数密度
    N_avg = np.mean(N_list)
    V_avg = np.mean(V_list)
    rho = N_avg / V_avg  # 平均数密度

    # 归一化：将 counts 转成 g(r)
    # 每个 bin 的壳体体积：4π r^2 dr
    shell_volumes = 4.0 * np.pi * (r_centers**2) * dr

    # 理想气体中，每帧每个粒子在壳体中的期望邻居数：
    # shell_volumes * rho
    # 总归一化因子 ~ n_frames_used * N_avg * shell_volumes * rho
    norm = n_frames_used * N_avg * shell_volumes * rho

    # 避免除零
    norm[norm == 0] = np.nan
    g_r = g_counts / norm

    return r_centers, g_r

def compute_structure_factor(df, q_max=0.5, dq=0.01, min_particles=20):
    """
    计算 3D 昆虫群的结构因子 S(q)，使用直接对粒子傅里叶变换的方法。

    参数：
        df : DataFrame，需包含 x,y,z,t
        q_max : 最大 q 值（单位与坐标的倒数一致）
        dq : q 的分辨率
        min_particles : 跳过粒子太少的帧

    返回：
        q_vals : 一维 q 数组
        S_q : 对应的 S(q)
    """

    df_pos = df[["x","y","z","t"]].dropna()

    # 取 q 的范围
    q_vals = np.arange(0.0, q_max, dq)
    S_q = np.zeros_like(q_vals)

    frames = np.sort(df_pos["t"].unique())
    n_frames_used = 0

    for t in frames:
        frame = df_pos[df_pos["t"] == t]
        coords = frame[["x","y","z"]].values
        N = coords.shape[0]
        if N < min_particles:
            continue

        n_frames_used += 1

        # 构建一批随机方向的 q-向量（保证各向同性平均）
        n_directions = 20
        dirs = np.random.normal(size=(n_directions, 3))
        dirs = dirs / np.linalg.norm(dirs, axis=1)[:,None]

        # 循环 q 值
        for iq, q in enumerate(q_vals):
            S_val = 0

            for d in dirs:
                q_vec = q * d
                phase = coords @ q_vec  # x*qx + y*qy + z*qz
                rho_q = np.exp(-1j * phase).sum()
                S_val += np.abs(rho_q)**2 / N

            S_q[iq] += S_val / n_directions

    if n_frames_used == 0:
        raise RuntimeError("没有适合计算的帧。")

    S_q /= n_frames_used
    return q_vals, S_q

def compute_velocity_structure_factor(df, q_max=0.5, dq=0.01, min_particles=20):
    """
    计算昆虫群的 velocity structure factor S_v(q)
    输入必须包含列: x,y,z,vx,vy,vz,t
    """
    dfv = df[["x","y","z","vx","vy","vz","t"]].dropna()

    # 构建 q 网格
    q_vals = np.arange(0.0, q_max, dq)
    S_vq = np.zeros_like(q_vals)

    # 用多方向平均减少噪声
    n_dirs = 20
    dirs = np.random.normal(size=(n_dirs,3))
    dirs = dirs / np.linalg.norm(dirs,axis=1)[:,None]

    frames = np.sort(dfv["t"].unique())
    n_frames_used = 0

    for t in frames:
        frame = dfv[dfv["t"]==t]
        coords = frame[["x","y","z"]].values
        vel = frame[["vx","vy","vz"]].values
        N = coords.shape[0]
        if N < min_particles:
            continue

        n_frames_used += 1

        for iq, q in enumerate(q_vals):
            Sv_temp = 0
            for d in dirs:
                q_vec = q * d
                phase = coords @ q_vec
                # velocity Fourier transform
                vq = (vel * np.exp(-1j * phase)[:,None]).sum(axis=0)
                Sv_temp += (np.abs(vq)**2).sum() / N
            S_vq[iq] += Sv_temp / n_dirs

    if n_frames_used == 0:
        raise RuntimeError("No valid frames for S_v(q).")

    S_vq /= n_frames_used
    return q_vals, S_vq
