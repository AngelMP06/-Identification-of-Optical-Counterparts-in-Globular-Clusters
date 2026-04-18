import numpy as np


def detect_msto(y_vals, x_smooth, bins_division, ymin, MS_index_start):
    derivative = np.diff(x_smooth)

    negative = np.where(derivative <= 0)[0]

    if len(negative) == 0:
        return 0, y_vals[0]

    msto_index = negative[np.argmax(np.diff(np.append(negative, negative[-1])))]

    offset = 1 / (2 * bins_division)

    msto_y = (msto_index + MS_index_start) / bins_division + ymin + offset

    return msto_index, msto_y