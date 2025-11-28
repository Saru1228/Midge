# swarm/hydrodynamic_derivatives.py

import numpy as np

def get_spacings(x_centers, y_centers, z_centers):
    """从网格中心坐标估计 dx, dy, dz（假设等间距）"""
    dx = float(np.mean(np.diff(x_centers)))
    dy = float(np.mean(np.diff(y_centers)))
    dz = float(np.mean(np.diff(z_centers)))
    return dx, dy, dz

def gradient_scalar(field, x_centers, y_centers, z_centers):
    """
    标量场梯度 ∇f，返回 (df/dx, df/dy, df/dz)，每个都是 3D array。
    field: 形状 (Nx, Ny, Nz)
    """
    dx, dy, dz = get_spacings(x_centers, y_centers, z_centers)
    dfdx, dfdy, dfdz = np.gradient(field, dx, dy, dz, edge_order=2)
    return dfdx, dfdy, dfdz

def divergence(vx, vy, vz, x_centers, y_centers, z_centers):
    """
    向量场散度 div v = dvx/dx + dvy/dy + dvz/dz
    vx,vy,vz: 3D array
    """
    dvx_dx, dvx_dy, dvx_dz = gradient_scalar(vx, x_centers, y_centers, z_centers)
    dvy_dx, dvy_dy, dvy_dz = gradient_scalar(vy, x_centers, y_centers, z_centers)
    dvz_dx, dvz_dy, dvz_dz = gradient_scalar(vz, x_centers, y_centers, z_centers)
    div = dvx_dx + dvy_dy + dvz_dz
    return div

def curl(vx, vy, vz, x_centers, y_centers, z_centers):
    """
    向量场旋度 curl v，返回 (ωx, ωy, ωz) 三个 3D array。
    """
    dvx_dx, dvx_dy, dvx_dz = gradient_scalar(vx, x_centers, y_centers, z_centers)
    dvy_dx, dvy_dy, dvy_dz = gradient_scalar(vy, x_centers, y_centers, z_centers)
    dvz_dx, dvz_dy, dvz_dz = gradient_scalar(vz, x_centers, y_centers, z_centers)

    # 旋度分量
    wx = dvz_dy - dvy_dz
    wy = dvx_dz - dvz_dx
    wz = dvy_dx - dvx_dy
    return wx, wy, wz

def laplacian_scalar(field, x_centers, y_centers, z_centers):
    """
    标量场拉普拉斯 ∇²f，用两次梯度近似。
    """
    dfdx, dfdy, dfdz = gradient_scalar(field, x_centers, y_centers, z_centers)
    d2fdx2, _, _ = gradient_scalar(dfdx, x_centers, y_centers, z_centers)
    _, d2fdy2, _ = gradient_scalar(dfdy, x_centers, y_centers, z_centers)
    _, _, d2fdz2 = gradient_scalar(dfdz, x_centers, y_centers, z_centers)
    return d2fdx2 + d2fdy2 + d2fdz2
