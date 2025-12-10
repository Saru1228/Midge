'''
centralize for each frame
calculate violocity accleration and jerk
'''
import numpy as np
import pandas as pd

def center_by_com_each_frame(df):
    """
    逐帧中心化，每个时间 t 减去群的质心位置。
    """
    df2 = df.copy()
    grouped = df2.groupby("t")
    df2["x"] = df2["x"] - grouped["x"].transform("mean")
    df2["y"] = df2["y"] - grouped["y"].transform("mean")
    df2["z"] = df2["z"] - grouped["z"].transform("mean")
    return df2

def add_speed_accel(df):
    """
    计算速度、加速度以及模长。
    """
    df2 = df.copy()
    for c in ["x", "y", "z"]:
        df2[f"v{c}"] = df2.groupby("id")[c].diff()
    for c in ["vx", "vy", "vz"]:
        df2[f"a{c[1]}"] = df2.groupby("id")[c].diff()

    df2["speed"] = np.sqrt(df2["vx"]**2 + df2["vy"]**2 + df2["vz"]**2)
    df2["accel"] = np.sqrt(df2["ax"]**2 + df2["ay"]**2 + df2["az"]**2)
    return df2

def add_jerk(df, dt=0.01):
    """
    Compute physical jerk = d(a)/dt using centered difference:
        j(t_i) = (a(t_{i+1}) - a(t_{i-1})) / (2*dt)

    Parameters
    ----------
    df : DataFrame
        Must contain columns ['id','t','ax','ay','az'].
    dt : float
        Time step, default 0.01 seconds.

    Returns
    -------
    df2 : DataFrame
        Adds columns jx_clean, jy_clean, jz_clean, jerk_clean.
    """
    df2 = df.copy()

    for c in ["ax", "ay", "az"]:
        df2[f"j{c[1]}"] = (
            df2.groupby("id")[c].shift(-1) - df2.groupby("id")[c].shift(+1)
        ) / (2 * dt)

    # Compute magnitude
    df2["jerk"] = np.sqrt(
        df2["jx"]**2 +
        df2["jy"]**2 +
        df2["jz"]**2
    )

    return df2



def preprocess_full(df):
    """
    完整预处理：中心化 → 速度 → 加速度 → jerk
    """
    df = center_by_com_each_frame(df)
    df = add_speed_accel(df)
    df = add_jerk(df)
    return df
