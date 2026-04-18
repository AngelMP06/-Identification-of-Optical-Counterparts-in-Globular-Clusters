import numpy as np
from scipy.signal import savgol_filter


def smooth_sequence(min_vals, max_vals, center_vals, bins_division, xmin):
    min_vals = np.array(min_vals)
    max_vals = np.array(max_vals)
    center_vals = np.array(center_vals)

    window = max(5, int(len(min_vals)/3) | 1)  # ensure odd
    poly = 3

    return (
        savgol_filter(min_vals, window, poly),
        savgol_filter(max_vals, window, poly),
        savgol_filter(center_vals, window, poly)
    )