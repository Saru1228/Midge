'''
smooth signal

slice_time_window
'''
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def smooth_signal(arr, window=9, poly=3):
    """
    使用 Savitzky–Golay 滤波器平滑轨迹。
    """
    return savgol_filter(arr, window_length=window, polyorder=poly)

def apply_smoothing(df, cols=["x", "y", "z"]):
    """
    对坐标做平滑。
    """
    df2 = df.copy()
    for c in cols:
        df2[c] = df2.groupby("id")[c].transform(lambda x: smooth_signal(x))
    return df2

def slice_time_window(df, window=4.0):
    """
    固定时间窗口切割（例如 4 秒）。
    """
    tmin = df["t"].min()
    tmax = df["t"].max()
    if tmax - tmin <= window:
        return df
    return df[df["t"] < tmin + window]

'''merge each dfs to a large DF'''
def merge_dict_of_dfs(dfs: dict):
    """
    输入: {"df1": df, "df2": df, ...}
    输出: 单个大 DataFrame
    """
    out = []
    for key in dfs:
        df = dfs[key].copy()
        df["source"] = key  # 记录来自哪个文件
        out.append(df)
    return pd.concat(out, ignore_index=True)
