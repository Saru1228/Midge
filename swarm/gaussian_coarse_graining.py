# gaussian_coarse_graining.py (KDTree version)
import numpy as np
from scipy.spatial import cKDTree

class GaussianCoarseGrainer:
    """
    Efficient Gaussian kernel coarse-graining using KDTree.
    Avoids full distance matrices and supports large datasets.
    """

    def __init__(self, grid_size=20.0, padding=10.0, cutoff_factor=3.0):
        """
        grid_size: 网格 mm
        padding: 边界拓展 mm
        cutoff_factor: Gaussian 截断半径系数（一般取 3σ）
        """
        self.grid_size = grid_size
        self.padding = padding
        self.cutoff_factor = cutoff_factor
        self.grid = None

    # ---------------------------------------------------------
    # 1. build grid
    # ---------------------------------------------------------
    def build_grid(self, df):
        x = df["x"].values
        y = df["y"].values
        z = df["z"].values

        xmin, xmax = x.min() - self.padding, x.max() + self.padding
        ymin, ymax = y.min() - self.padding, y.max() + self.padding
        zmin, zmax = z.min() - self.padding, z.max() + self.padding

        x_edges = np.arange(xmin, xmax + self.grid_size, self.grid_size)
        y_edges = np.arange(ymin, ymax + self.grid_size, self.grid_size)
        z_edges = np.arange(zmin, zmax + self.grid_size, self.grid_size)

        self.x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        self.y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        self.z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

        self.grid = (self.x_centers, self.y_centers, self.z_centers)

        print("Grid:", len(self.x_centers), len(self.y_centers), len(self.z_centers))
        return self.grid

    # ---------------------------------------------------------
    # 2. Gaussian kernel
    # ---------------------------------------------------------
    def _gaussian(self, r, sigma):
        return np.exp(-0.5 * (r / sigma)**2)

    # ---------------------------------------------------------
    # 3. General coarse-graining for scalar field
    # ---------------------------------------------------------
    def _coarse_scalar(self, df, values, sigma):

        coords = df[["x","y","z"]].values
        tree = cKDTree(coords)

        Nx, Ny, Nz = len(self.x_centers), len(self.y_centers), len(self.z_centers)
        field = np.zeros((Nx, Ny, Nz))
        weight_sum = np.zeros_like(field)

        cutoff = self.cutoff_factor * sigma

        for ix, xc in enumerate(self.x_centers):
            for iy, yc in enumerate(self.y_centers):
                for iz, zc in enumerate(self.z_centers):

                    # query particles in 3σ radius
                    idx = tree.query_ball_point([xc,yc,zc], r=cutoff)
                    if len(idx) == 0:
                        field[ix,iy,iz] = np.nan
                        continue

                    pts = coords[idx]
                    vals = values[idx]

                    dist = np.linalg.norm(pts - np.array([xc,yc,zc]), axis=1)
                    W = self._gaussian(dist, sigma)

                    wsum = W.sum()
                    if wsum == 0:
                        field[ix,iy,iz] = np.nan
                    else:
                        field[ix,iy,iz] = np.sum(W * vals) / wsum

        return field

    # ---------------------------------------------------------
    # 4. Density
    # ---------------------------------------------------------
    def coarse_density_sigma(self, df, sigma):
        ones = np.ones(len(df))
        return self._coarse_scalar(df, ones, sigma)

    # ---------------------------------------------------------
    # 5. Velocity
    # ---------------------------------------------------------
    def coarse_velocity_sigma(self, df, sigma):
        vx = self._coarse_scalar(df, df["vx"].values, sigma)
        vy = self._coarse_scalar(df, df["vy"].values, sigma)
        vz = self._coarse_scalar(df, df["vz"].values, sigma)
        return vx, vy, vz

    # ---------------------------------------------------------
    # 6. Accel
    # ---------------------------------------------------------
    def coarse_accel_sigma(self, df, sigma):
        ax = self._coarse_scalar(df, df["ax"].values, sigma)
        ay = self._coarse_scalar(df, df["ay"].values, sigma)
        az = self._coarse_scalar(df, df["az"].values, sigma)
        return ax, ay, az

    # ---------------------------------------------------------
    # 7. Jerk
    # ---------------------------------------------------------
    def coarse_jerk_sigma(self, df, sigma):
        jx = self._coarse_scalar(df, df["jx"].values, sigma)
        jy = self._coarse_scalar(df, df["jy"].values, sigma)
        jz = self._coarse_scalar(df, df["jz"].values, sigma)
        return jx, jy, jz
