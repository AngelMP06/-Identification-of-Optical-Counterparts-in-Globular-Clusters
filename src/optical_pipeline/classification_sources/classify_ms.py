import numpy as np
import pandas as pd
from src.models import MainSequenceResults

from ..utils.colors import get_color_columns


def classify_position_MS(
    df: pd.DataFrame,
    CMD_config: int,
    main_sequence_results: MainSequenceResults
    ) -> pd.DataFrame:
    """
    Classify sources based on their position relative to the Main Sequence (MS).

    - The classification is done segment-by-segment using interpolated MS boundaries.

    - The classification also depends on the position with respect to the MSTO and SGB
    """

    # --- Copy to avoid modifying original ---
    df = df.copy()


    # --- Get color columns ---
    Color1, Color2 = get_color_columns(CMD_config)

    x_col = f"{Color1} - {Color2}"
    y_col = f"{Color2}_Mag"
    pos_col = f"position_{CMD_config}"

    df[pos_col] = "Unknown"

    x_vals = df[x_col].values
    y_vals = df[y_col].values

    width_MS = main_sequence_results.max_smooth - main_sequence_results.min_smooth
    mean_width_MS = 2 * np.mean(width_MS)


    # --- Loop over MS segments ---
    for i in range(len(main_sequence_results.main_sequence_y) - 1):

        xmin_1, xmin_2 = main_sequence_results.min_smooth[i], main_sequence_results.min_smooth[i + 1]
        xmax_1, xmax_2 = main_sequence_results.max_smooth[i], main_sequence_results.max_smooth[i + 1]
        y1, y2 = main_sequence_results.main_sequence_y[i], main_sequence_results.main_sequence_y[i + 1]

        slope = (xmin_2 - xmin_1) / (y2 - y1)

        mask = (y_vals >= y1) & (y_vals < y2)

        if not np.any(mask):
            continue

        y_seg = y_vals[mask]
        x_seg = x_vals[mask]


        # --- Interpolate MS boundaries ---
        x_min_real = xmin_1 + (y_seg - y1) * slope
        x_max_real = xmax_1 + (y_seg - y1) * slope


        # --- Define thresholds ---
        conditions = [
            x_seg < x_min_real - mean_width_MS,
            x_seg < x_min_real,
            x_seg < x_max_real,
            x_seg < x_max_real + mean_width_MS,
        ]


        # --- Choose labels depending on region ---
        if i < main_sequence_results.index_SGB:
            labels = [
                "bluer than MS L2",
                "bluer than SGB L1",
                "SGB",
                "S-SGB",
            ]

        elif i < main_sequence_results.index_MSTO:
            labels = [
                "bluer than MS L2",
                "bluer than MSTO L1",
                "MSTO",
                "S-SGB",
            ]

        else:
            labels = [
                "bluer than MS L2",
                "bluer than MS L1",
                "MS",
                "redder than MS L1",
            ]

        result = np.select(conditions, labels, default="redder than MS L2")

        df.loc[mask, pos_col] = result


    # --- Below MS ---
    df.loc[y_vals > main_sequence_results.main_sequence_y[-1], pos_col] = "below the MS"

    return df