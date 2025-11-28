'''
compute trajact lenth of each insects
gyration radius
energy and 
PCA of velocity
'''
import numpy as np
import pandas as pd

def track_lengths(df):
    """
    计算每只昆虫的轨迹长度（时间长度）。
    """
    return df.groupby("id")["t"].agg(["min", "max", "count"])

def gyration_radius(df):
    """
    群体大小的度量：gyration radius。
    """
    x2 = (df["x"]**2 + df["y"]**2 + df["z"]**2).mean()
    return np.sqrt(x2)

def compute_relative_kinetic_energy(df):
    """
    相对动能，用于衡量群体整体的运动活性。
    """
    return (df["speed"]**2).mean()

def velocity_pca(df):
    """
    对速度场做 PCA，检查群体运动是否具有主模态。
    """
    V = df[["vx", "vy", "vz"]].dropna().values
    C = np.cov(V.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    return eigvals, eigvecs
