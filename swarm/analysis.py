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
