import numpy as np
from scipy.signal import savgol_filter


def smooth_sequence(normalized_min: np.ndarray, 
                    normalized_max: np.ndarray, 
                    normalized_center: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect the Main Sequence Turn-Off (MSTO) point from the smoothed
    main sequence center line.

    The MSTO is identified as the point where the main sequence stops
    decreasing in magnitude and begins to turn toward the red (i.e.,
    where the slope of the sequence changes sign).
    """

    normalized_min = np.array(normalized_min)
    normalized_max = np.array(normalized_max)
    normalized_center = np.array(normalized_center)

    window = max(5, int(len(normalized_min)/3) | 1) # Window size for Savitzky-Golay filter (must be odd)

    poly = min(7, window - 1)

    min_smooth = np.array(savgol_filter(normalized_min, window, poly))
    max_smooth = np.array(savgol_filter(normalized_max, window, poly))
    center_smooth = np.array(savgol_filter(normalized_center, window, poly))

    return min_smooth, max_smooth, center_smooth