# ================================================
# Cell 2: Define MEM helper functions
# ================================================

import numpy as np

# -------------------------------
# empirical pair correlation C_data
# -------------------------------
def compute_C_data(s, neighbor_list):
    dots = []
    for i in range(len(s)):
        for j in neighbor_list[i]:
            dots.append(np.dot(s[i], s[j]))
    return np.mean(dots)

# -------------------------------
# compute polarization m
# -------------------------------
def compute_polarization(s):
    S = np.sum(s, axis=0)
    return np.linalg.norm(S) / len(s)

# -------------------------------
# mean-field prediction: C_model(J)
# Bialek 2012 PNAS Supplement
# -------------------------------
def C_model(J, m):
    x = J*m
    if x == 0:
        return 0
    return (np.cosh(x)/np.sinh(x)) - (1/x)   # coth(x) - 1/x

# consistency equation F(J) = 0
def F(J, C_data, m):
    return C_model(J, m) - C_data

