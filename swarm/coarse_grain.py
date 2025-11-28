'''coarse-grain field
01 density field
02 velocity field
03 accerlrate field
04 jerd feld
'''
import numpy as np

def compute_density_field(df, grid_size=5.0, padding=1.0):
    """
    使用合理的 coarse-graining 网格计算密度场 ρ(r)。
    单位：假设原始坐标是 mm。

    grid_size=x.0 → 每个网格 x mm
    padding=x.0 → 上下预留 x mm
    """
    x = df["x"].values
    y = df["y"].values
    z = df["z"].values

    # 自动设置边界
    xmin, xmax = x.min() - padding, x.max() + padding
    ymin, ymax = y.min() - padding, y.max() + padding
    zmin, zmax = z.min() - padding, z.max() + padding

    # 创建网格边界（减少内存）
    x_edges = np.arange(xmin, xmax + grid_size, grid_size)
    y_edges = np.arange(ymin, ymax + grid_size, grid_size)
    z_edges = np.arange(zmin, zmax + grid_size, grid_size)

    print("Grid sizes:", len(x_edges), len(y_edges), len(z_edges))

    # 计算密度 (counts per grid cell)
    rho, edges = np.histogramdd(
        sample=np.vstack([x, y, z]).T,
        bins=(x_edges, y_edges, z_edges)
    )

    return rho, edges

def make_grid_from_df(df, grid_size=5.0, padding=5.0):
    """
    根据一段数据 df 自动构建 3D 网格边界。
    假设坐标单位为 mm。
    返回: (x_edges, y_edges, z_edges)
    """
    x = df["x"].values
    y = df["y"].values
    z = df["z"].values

    xmin, xmax = x.min() - padding, x.max() + padding
    ymin, ymax = y.min() - padding, y.max() + padding
    zmin, zmax = z.min() - padding, z.max() + padding

    x_edges = np.arange(xmin, xmax + grid_size, grid_size)
    y_edges = np.arange(ymin, ymax + grid_size, grid_size)
    z_edges = np.arange(zmin, zmax + grid_size, grid_size)

    print("Grid size (Nx, Ny, Nz):", len(x_edges)-1, len(y_edges)-1, len(z_edges)-1)
    return x_edges, y_edges, z_edges


def coarse_grain_velocity_frame(df_frame, x_edges, y_edges, z_edges):
    """
    对单帧数据 df_frame 进行 coarse-grain，得到:
    - rho: 每个体素的粒子数（可以除以体积变为密度）
    - vx_field, vy_field, vz_field: 每个格点上的平均速度场

    df_frame 需要包含: x,y,z,vx,vy,vz
    """
    coords = df_frame[["x", "y", "z"]].values
    vx = df_frame["vx"].values
    vy = df_frame["vy"].values
    vz = df_frame["vz"].values

    # 粒子数（top-hat 核）
    rho, edges = np.histogramdd(
        sample=coords,
        bins=(x_edges, y_edges, z_edges)
    )

    # 各分量加权和
    vx_sum, _ = np.histogramdd(
        sample=coords,
        bins=(x_edges, y_edges, z_edges),
        weights=vx
    )
    vy_sum, _ = np.histogramdd(
        sample=coords,
        bins=(x_edges, y_edges, z_edges),
        weights=vy
    )
    vz_sum, _ = np.histogramdd(
        sample=coords,
        bins=(x_edges, y_edges, z_edges),
        weights=vz
    )

    # 避免除零：先复制一份
    rho_safe = rho.copy().astype(float)
    rho_safe[rho_safe == 0] = np.nan  # 没有粒子的格子设为 NaN

    vx_field = vx_sum / rho_safe
    vy_field = vy_sum / rho_safe
    vz_field = vz_sum / rho_safe

    return rho, (vx_field, vy_field, vz_field), edges

def coarse_grain_accel_frame(df_frame, x_edges, y_edges, z_edges):
    """
    对单帧数据 df_frame 进行 coarse-grain，得到:
    - rho: 每个体素的粒子数
    - ax_field, ay_field, az_field: 每个格点上的平均加速度场

    df_frame 需要包含: x,y,z,ax,ay,az
    """
    coords = df_frame[["x", "y", "z"]].values
    ax = df_frame["ax"].values
    ay = df_frame["ay"].values
    az = df_frame["az"].values

    rho, edges = np.histogramdd(
        sample=coords,
        bins=(x_edges, y_edges, z_edges)
    )

    ax_sum, _ = np.histogramdd(
        sample=coords,
        bins=(x_edges, y_edges, z_edges),
        weights=ax
    )
    ay_sum, _ = np.histogramdd(
        sample=coords,
        bins=(x_edges, y_edges, z_edges),
        weights=ay
    )
    az_sum, _ = np.histogramdd(
        sample=coords,
        bins=(x_edges, y_edges, z_edges),
        weights=az
    )

    rho_safe = rho.copy().astype(float)
    rho_safe[rho_safe == 0] = np.nan

    ax_field = ax_sum / rho_safe
    ay_field = ay_sum / rho_safe
    az_field = az_sum / rho_safe

    return rho, (ax_field, ay_field, az_field), edges

def coarse_grain_jerk_frame(df_frame, x_edges, y_edges, z_edges):
    """
    对单帧数据 df_frame 进行 coarse-grain，得到:
    - rho: 每个体素的粒子数
    - jx_field, jy_field, jz_field: 每个格点上的平均 jerk 场

    df_frame 需要包含: x,y,z,jx,jy,jz
    """
    coords = df_frame[["x", "y", "z"]].values
    jx = df_frame["jx"].values
    jy = df_frame["jy"].values
    jz = df_frame["jz"].values

    rho, edges = np.histogramdd(
        sample=coords,
        bins=(x_edges, y_edges, z_edges)
    )

    jx_sum, _ = np.histogramdd(
        sample=coords,
        bins=(x_edges, y_edges, z_edges),
        weights=jx
    )
    jy_sum, _ = np.histogramdd(
        sample=coords,
        bins=(x_edges, y_edges, z_edges),
        weights=jy
    )
    jz_sum, _ = np.histogramdd(
        sample=coords,
        bins=(x_edges, y_edges, z_edges),
        weights=jz
    )

    rho_safe = rho.copy().astype(float)
    rho_safe[rho_safe == 0] = np.nan

    jx_field = jx_sum / rho_safe
    jy_field = jy_sum / rho_safe
    jz_field = jz_sum / rho_safe

    return rho, (jx_field, jy_field, jz_field), edges