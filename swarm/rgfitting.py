import numpy as np
from scipy.ndimage import gaussian_filter

def laplacian_vector_field(a_field, dx):
    """
    Compute Laplacian of a 3D vector field a(x) on a regular grid.
    a_field: ndarray (Nx, Ny, Nz, 3)
    dx: grid spacing (assumed isotropic)
    Returns: lap_a with same shape as a_field
    """
    # For each vector component, compute scalar Laplacian
    lap_components = []
    for c in range(3):
        comp = a_field[..., c]
        # scalar Laplacian: sum of second derivatives along x,y,z
        d2 = 0
        for axis in (0, 1, 2):
            g1 = np.gradient(comp, dx, axis=axis)
            g2 = np.gradient(g1, dx, axis=axis)
            d2 += g2
        lap_components.append(d2)
    lap_a = np.stack(lap_components, axis=-1)
    return lap_a

def fit_D_gamma(a_field, j_field, dx, mask_boundary=True, boundary_width=1):
    """
    Fit effective parameters D(k), gamma(k) from
    j ≈ -gamma * a + D * laplacian(a) on a 3D grid.
    
    Parameters
    ----------
    a_field : ndarray (Nx, Ny, Nz, 3)
        Coarse-grained acceleration field at scale k.
    j_field : ndarray (Nx, Ny, Nz, 3)
        Coarse-grained jerk field at the same scale/time.
    dx : float
        Grid spacing.
    mask_boundary : bool
        Whether to discard a few boundary layers (derivatives noisy at edges).
    boundary_width : int
        How many cells to cut from each boundary if mask_boundary is True.
        
    Returns
    -------
    D_hat : float
    gamma_hat : float
    """
    # 1. Compute Laplacian of a
    lap_a = laplacian_vector_field(a_field, dx)
    
    # 2. Optionally cut boundary to avoid derivative artifacts
    if mask_boundary:
        sl = slice(boundary_width, -boundary_width)
        a_cut   = a_field[sl, sl, sl, :]
        j_cut   = j_field[sl, sl, sl, :]
        lap_cut = lap_a[sl, sl, sl, :]
    else:
        a_cut   = a_field
        j_cut   = j_field
        lap_cut = lap_a

    # 3. Flatten all spatial positions and vector components into 1D arrays
    a_flat   = a_cut.reshape(-1, 3)
    j_flat   = j_cut.reshape(-1, 3)
    lap_flat = lap_cut.reshape(-1, 3)
    
    # 只用有限数量的点（如果你想降采样，否则就用全部）
    # 这里默认用全部：
    # 构造标量样本：对 3 个分量都加入拟合
    a_scalar   = a_flat.ravel()      # shape (N*3,)
    j_scalar   = j_flat.ravel()
    lap_scalar = lap_flat.ravel()
    
    # 4. 构建设计矩阵 X 和目标 y
    # j = -gamma * a + D * lap(a)
    X = np.column_stack([-a_scalar, lap_scalar])  # shape (N*3, 2)
    y = j_scalar                                  # shape (N*3,)
    
    # 5. 最小二乘拟合 [gamma, D]
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    gamma_hat, D_hat = coeffs  # 注意顺序与 X 对应 [-a, lap_a]
    
    return D_hat, gamma_hat

def gaussian_smooth_field(field, sigma, mode='reflect'):
    """
    Apply Gaussian smoothing to a 3D scalar or vector field.

    Parameters
    ----------
    field : np.ndarray
        Shape either:
            (Nx, Ny, Nz)               - scalar field
        or  (Nx, Ny, Nz, 3)            - vector field (v_x, v_y, v_z)

    sigma : float
        Standard deviation of Gaussian kernel (in grid units).
        You can set sigma = k for RG coarse-grain at scale k.

    mode : str
        Boundary handling mode for scipy.ndimage.gaussian_filter.
        Default 'reflect'.

    Returns
    -------
    smoothed_field : np.ndarray
        Smoothed field with same shape as input.
    """
    field = np.asarray(field)

    # -------- Case 1: scalar field --------
    if field.ndim == 3:
        # Replace NaN with local mean (0) for stability
        field_clean = np.nan_to_num(field, nan=0.0)
        smoothed = gaussian_filter(field_clean, sigma=sigma, mode=mode)
        return smoothed

    # -------- Case 2: vector field --------
    elif field.ndim == 4 and field.shape[-1] == 3:
        smoothed = np.zeros_like(field)
        for i in range(3):
            fi = np.nan_to_num(field[..., i], nan=0.0)
            smoothed[..., i] = gaussian_filter(fi, sigma=sigma, mode=mode)
        return smoothed

    else:
        raise ValueError(
            f"gaussian_smooth_field: unsupported field shape {field.shape}. "
            "Expected (Nx,Ny,Nz) or (Nx,Ny,Nz,3)."
        )
import numpy as np
from sklearn.neighbors import NearestNeighbors

def topo_laplacian_vector(pos, vec, k):
    """
    Compute a topological Laplacian of a vector field defined on particles.

    Parameters
    ----------
    pos : (N,3) array
        Particle positions (x,y,z) for a single time window (or slice).
    vec : (N,3) array
        Vector attached to each particle (e.g. acceleration a or jerk j).
    k : int
        Number of topological neighbours (KNN).

    Returns
    -------
    lap_vec : (N,3) array
        Topological Laplacian of vec at each particle:
            L_topo vec_i = mean_{j in N_k(i)} vec_j - vec_i
    """
    pos = np.asarray(pos)
    vec = np.asarray(vec)

    if pos.shape[0] != vec.shape[0]:
        raise ValueError(f"pos and vec must have same length, got {pos.shape[0]} vs {vec.shape[0]}.")

    N = pos.shape[0]
    if N <= k:
        raise ValueError(f"Number of points N={N} must be > k={k}.")

    # 建 k+1 个邻居（包括自己），再把自己去掉
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(pos)
    _, indices = nbrs.kneighbors(pos)

    lap_vec = np.zeros_like(vec)

    for i in range(N):
        neigh_idx = indices[i, 1:]   # 去掉 indices[i,0] = i 自己
        neigh_mean = vec[neigh_idx].mean(axis=0)
        lap_vec[i] = neigh_mean - vec[i]

    return lap_vec
def fit_D_gamma_topological(
    pos,
    a_vec,
    j_vec,
    k,
    min_norm_threshold=1e-8
):
    """
    Fit D(k) and gamma(k) from topological Laplacian relation:
        j ≈ D * L_topo(a) - gamma * a

    Parameters
    ----------
    pos : (N,3) array
        Particle positions.
    a_vec : (N,3) array
        Acceleration vectors a_i.
    j_vec : (N,3) array
        Jerk vectors j_i.
    k : int
        Topological neighbour number for Laplacian.
    min_norm_threshold : float
        Threshold to remove nearly-zero rows to avoid dividing by noise.

    Returns
    -------
    D_hat : float
        Fitted D(k).
    gamma_hat : float
        Fitted gamma(k).
    """
    pos = np.asarray(pos)
    a_vec = np.asarray(a_vec)
    j_vec = np.asarray(j_vec)

    # 统一去掉 NaN
    mask = (~np.isnan(a_vec).any(axis=1)) & (~np.isnan(j_vec).any(axis=1))
    pos = pos[mask]
    a_vec = a_vec[mask]
    j_vec = j_vec[mask]

    if pos.shape[0] <= k + 1:
        raise ValueError(f"Not enough points after masking: N={pos.shape[0]}, k={k}")

    # 计算 topological Laplacian of a
    Lap_a = topo_laplacian_vector(pos, a_vec, k=k)   # (N,3)

    # 展平：把所有分量堆成一维
    L_flat = Lap_a.reshape(-1)       # 第一列
    a_flat = a_vec.reshape(-1)       # 第二列
    j_flat = j_vec.reshape(-1)       # 目标

    # 去掉都接近 0 的点，避免纯噪音主导拟合
    mag = np.sqrt(L_flat**2 + a_flat**2)
    good = mag > min_norm_threshold

    X = np.vstack([L_flat[good], -a_flat[good]]).T  # shape (M,2)
    y = j_flat[good]

    if X.shape[0] < 10:
        raise ValueError(f"Too few effective samples to fit D,gamma: M={X.shape[0]}")

    # 最小二乘拟合： y ≈ [D, gamma]^T · [L, -a]
    # 即 j ≈ D * Lap_a - gamma * a
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    D_hat, gamma_hat = coef[0], coef[1]

    return float(D_hat), float(gamma_hat)
def scan_D_gamma_topological(
    df_win,
    k_list,
    pos_cols=('x','y','z'),
    a_cols=('ax','ay','az'),
    j_cols=('jx','jy','jz')
):
    """
    对给定时间窗口 df_win，扫描一系列 topological k，
    拟合每个 k 下的 D(k), gamma(k)。

    Parameters
    ----------
    df_win : pandas.DataFrame
        已选好的时间窗口数据（例如 1s 内的所有点）。
    k_list : list[int]
        要扫描的 KNN 邻居数列表。
    pos_cols : tuple[str]
        位置列名。
    a_cols : tuple[str]
        加速度列名。
    j_cols : tuple[str]
        jerk 列名。

    Returns
    -------
    results : dict
        包含:
        {
          'k_list': [...],
          'D_list': [...],
          'gamma_list': [...],
        }
    """
    import numpy as np

    pos = df_win[list(pos_cols)].values
    a_vec = df_win[list(a_cols)].values
    j_vec = df_win[list(j_cols)].values

    D_list = []
    gamma_list = []
    valid_k = []

    for k in k_list:
        try:
            D_k, gamma_k = fit_D_gamma_topological(pos, a_vec, j_vec, k=k)
            D_list.append(D_k)
            gamma_list.append(gamma_k)
            valid_k.append(k)
            print(f"k={k}: D={D_k:.3g}, gamma={gamma_k:.3g}")
        except Exception as e:
            print(f"k={k}: failed -> {e}")

    return {
        'k_list': np.array(valid_k),
        'D_list': np.array(D_list),
        'gamma_list': np.array(gamma_list)
    }
