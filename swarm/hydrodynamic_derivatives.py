# swarm/hydrodynamic_derivatives.py

import numpy as np
from numpy.fft import fftn, fftshift
import scipy.ndimage as nd

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

def gradient_energy(a_field, dx):
    """
    Compute gradient energy E = <|∇a|^2>
    a_field: ndarray (Nx, Ny, Nz, 3)
    dx: grid spacing
    """
    # Compute spatial gradients along x,y,z
    grad = np.gradient(a_field, dx, axis=(0,1,2))  # returns list of 3 arrays
    
    # grad[i] has shape (Nx,Ny,Nz,3)
    # Compute |∂_i a|^2 for each direction i
    grad_sq = [(g**2).sum(axis=-1) for g in grad]  # sum over vector components
    
    # Total |∇a|^2 = |∂x a|^2 + |∂y a|^2 + |∂z a|^2
    grad_energy = sum(grad_sq)
    
    return grad_energy.mean() 

def compute_velocity_correlation(vx, vy, vz, x_centers, y_centers, z_centers, max_dist=None, nbins=30):

    # 3D positions of grid centers
    X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")

    # flatten fields
    v_flat = np.vstack([
        vx.flatten(), 
        vy.flatten(), 
        vz.flatten()
    ]).T

    pos_flat = np.vstack([
        X.flatten(), 
        Y.flatten(), 
        Z.flatten()
    ]).T

    # remove NaN voxels
    mask = ~np.isnan(v_flat).any(axis=1)
    v_flat = v_flat[mask]
    pos_flat = pos_flat[mask]

    N = len(v_flat)
    print(f"Using {N} non-NaN voxels for C_v(r)")

    # pairwise distances
    # for large grids, subsample
    if N > 5000:
        idx = np.random.choice(N, 4000, replace=False)
        pos_flat = pos_flat[idx]
        v_flat = v_flat[idx]
        N = len(v_flat)
        print(f"Subsampled to {N} voxels")

    # compute pairwise dot products
    diff = pos_flat[:, None, :] - pos_flat[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    vdot = np.sum(v_flat[:, None, :] * v_flat[None, :, :], axis=2)

    # bins
    if max_dist is None:
        max_dist = np.max(dist)

    bins = np.linspace(0, max_dist, nbins+1)
    rvals = 0.5*(bins[:-1] + bins[1:])
    C = np.zeros(nbins)

    for i in range(nbins):
        mask2 = (dist >= bins[i]) & (dist < bins[i+1])
        if np.sum(mask2) > 0:
            C[i] = np.mean(vdot[mask2])

    return rvals, C


def compute_structure_factor_3D(rho):
    # replace NaN with 0
    rho_filled = np.nan_to_num(rho, nan=0.0)

    rho_k = fftshift(fftn(rho_filled))
    S_k = np.abs(rho_k)**2

    # 3D k-grid
    Nx, Ny, Nz = rho.shape
    kx = np.fft.fftfreq(Nx)
    ky = np.fft.fftfreq(Ny)
    kz = np.fft.fftfreq(Nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")

    k_mag = np.sqrt(KX**2 + KY**2 + KZ**2)

    # isotropic average
    kbins = np.linspace(0, np.max(k_mag), 40)
    kvals = 0.5*(kbins[:-1] + kbins[1:])
    S_avg = np.zeros_like(kvals)

    for i in range(len(kvals)):
        mask = (k_mag >= kbins[i]) & (k_mag < kbins[i+1])
        if np.sum(mask) > 0:
            S_avg[i] = np.mean(S_k[mask])

    return kvals, S_avg

def compute_S_v_qw(v_fields, dt):
    """
    v_fields: list of v(x,y,z) arrays for each time t
    dt: time between frames
    """
    v_fields = np.array(v_fields)  # shape (T, Nx,Ny,Nz,3)
    T = v_fields.shape[0]

    # temporal FFT
    v_w = np.fft.fftn(v_fields, axes=[0])

    # spatial FFT per frequency
    v_qw = np.fft.fftn(v_w, axes=[1,2,3])

    # power spectrum
    S = np.abs(v_qw)**2
    return S
    print("Sound mode module ready. Feed v(x,y,z,t) to compute S(q,ω).")

def laplacian_field(field, dx):
    """
    Compute the 3D Laplacian ∇² field for either:
      - scalar fields: (Nx, Ny, Nz)
      - vector fields: (Nx, Ny, Nz, 3)

    Uses:
      ∇² f ≈ (Σ_neighbors f - 6f) / dx^2 
    With Neumann boundary conditions (mode='nearest').

    Parameters
    ----------
    field : ndarray
        Scalar or vector field.
    dx : float
        Grid spacing.

    Returns
    -------
    lap : ndarray
        Laplacian with same shape as field.
    """

    field = np.asarray(field)

    # Case 1: scalar field
    if field.ndim == 3:
        return nd.laplace(field, mode="nearest") / (dx * dx)

    # Case 2: vector field (Nx,Ny,Nz,3)
    elif field.ndim == 4 and field.shape[-1] == 3:
        lap = np.zeros_like(field)
        for c in range(3):
            lap[..., c] = nd.laplace(field[..., c], mode="nearest") / (dx * dx)
        return lap

    else:
        raise ValueError(
            f"Unsupported field shape {field.shape}. "
            "Expect (Nx,Ny,Nz) or (Nx,Ny,Nz,3)."
        )
