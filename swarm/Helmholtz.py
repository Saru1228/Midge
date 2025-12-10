import numpy as np

def helmholtz_decompose_fft(field, dx):
    """
    对 3D 向量场做 Helmholtz 分解：
        f(x) = f_grad(x) + f_curl(x)
      其中：
        f_grad = irrotational 部分（~ 梯度势场）
        f_curl = solenoidal 部分（~ 旋度主导）

    参数:
      field: ndarray, shape = (Nx, Ny, Nz, 3)
      dx: 标量，网格间距（假设各向同性）

    返回:
      f_grad, f_curl, stats  三个对象
        f_grad, f_curl: 与 field 同形状
        stats: dict, 包含能量占比等
    说明:
      - 使用 FFT，假设周期边界（periodic BC）
      - 对 NaN 体素：在分解时视为 0，但能量统计时只在原本非 NaN 区域上计算
    """
    field = np.array(field, dtype=float)

    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError("field must have shape (Nx, Ny, Nz, 3)")

    Nx, Ny, Nz, _ = field.shape

    # ---- 1. 处理 NaN：计算时用 0 填充，但保留 mask 用于后面统计 ----
    mask_valid = ~np.isnan(field).any(axis=3)
    f = field.copy()
    f[~mask_valid] = 0.0

    fx = f[..., 0]
    fy = f[..., 1]
    fz = f[..., 2]

    # ---- 2. 计算散度 div f ----
    # 使用中心差分 np.gradient，步长为 dx
    div = (np.gradient(fx, dx, axis=0) +
           np.gradient(fy, dx, axis=1) +
           np.gradient(fz, dx, axis=2))

    # ---- 3. FFT 求解 Poisson 方程: ∇²φ = div ----
    div_k = np.fft.fftn(div)

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dx)
    kz = 2 * np.pi * np.fft.fftfreq(Nz, d=dx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")

    k2 = KX**2 + KY**2 + KZ**2

    phi_k = np.zeros_like(div_k, dtype=complex)
    nonzero = (k2 > 0)
    phi_k[nonzero] = -div_k[nonzero] / k2[nonzero]
    # k=0 模（整体常数）设为 0 即可

    phi = np.fft.ifftn(phi_k).real

    # ---- 4. 用谱方法计算 grad φ ----
    phi_k = np.fft.fftn(phi)
    grad_phi_kx = 1j * KX * phi_k
    grad_phi_ky = 1j * KY * phi_k
    grad_phi_kz = 1j * KZ * phi_k

    grad_x = np.fft.ifftn(grad_phi_kx).real
    grad_y = np.fft.ifftn(grad_phi_ky).real
    grad_z = np.fft.ifftn(grad_phi_kz).real

    f_grad = np.stack([grad_x, grad_y, grad_z], axis=3)
    f_curl = f - f_grad

    # ---- 5. 只在原始非 NaN 位置上统计能量 ----
    if np.any(mask_valid):
        f_valid      = f[mask_valid]          # (N_valid, 3)
        f_grad_valid = f_grad[mask_valid]     # (N_valid, 3)
        f_curl_valid = f_curl[mask_valid]     # (N_valid, 3)

        E_tot   = np.mean(np.sum(f_valid**2, axis=1))
        E_grad  = np.mean(np.sum(f_grad_valid**2, axis=1))
        E_curl  = np.mean(np.sum(f_curl_valid**2, axis=1))
    else:
        E_tot = E_grad = E_curl = np.nan

    stats = {
        "E_total": E_tot,
        "E_grad": E_grad,
        "E_curl": E_curl,
        "frac_grad": E_grad / E_tot if E_tot > 0 else np.nan,
        "frac_curl": E_curl / E_tot if E_tot > 0 else np.nan,
        "valid_ratio": mask_valid.mean()
    }

    return f_grad, f_curl, stats
