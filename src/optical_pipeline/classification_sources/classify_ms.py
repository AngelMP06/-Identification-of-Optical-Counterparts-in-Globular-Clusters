import numpy as np
from ..utils.colors import get_color_columns


def classify_position_MS(
    df,
    graphic_comp,
    main_sequence_y,
    min_smooth,
    max_smooth,
    index_SGB,
    index_MSTO,
    significantly_bluer_limit,
    bluer_limit,
    redder_limit,
    significantly_redder_limit
):
    """
    Classify stars based on their position relative to the Main Sequence (MS).

    - The classification is done segment-by-segment using interpolated MS boundaries.

    - The classification also depends on the position with respect to the MSTO and SGB
    """

    # --- Copy to avoid modifying original ---
    df = df.copy()

    # --- Get color columns ---
    Color1, Color2 = get_color_columns(graphic_comp)

    x_col = f"{Color1} - {Color2}"
    y_col = f"{Color2}_Mag"
    pos_col = f"position_{graphic_comp}"

    df[pos_col] = "Unknown"

    x_vals = df[x_col].values
    y_vals = df[y_col].values

    # --- Loop over MS segments ---
    for i in range(len(main_sequence_y) - 1):

        xmin_1, xmin_2 = min_smooth[i], min_smooth[i + 1]
        xmax_1, xmax_2 = max_smooth[i], max_smooth[i + 1]
        y1, y2 = main_sequence_y[i], main_sequence_y[i + 1]

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
            x_seg < x_min_real - significantly_bluer_limit,
            x_seg < x_min_real - bluer_limit,
            x_seg < x_min_real,
            x_seg < x_max_real,
            x_seg < x_max_real + redder_limit,
            x_seg < x_max_real + significantly_redder_limit,
        ]

        # --- Choose labels depending on region ---
        if i < index_SGB:
            labels = [
                "bluer than MS L3",
                "bluer than MS L2",
                "bluer than SGB",
                "Sub Giant Branch",
                "redder than SGB",
                "redder than MS L2",
            ]

        elif i < index_MSTO:
            labels = [
                "bluer than MS L3",
                "bluer than MS L2",
                "bluer than MSTO",
                "MSTO",
                "redder than MSTO",
                "redder than MS L2",
            ]

        else:
            labels = [
                "bluer than MS L3",
                "bluer than MS L2",
                "bluer than MS L1",
                "MS",
                "redder than MS L1",
                "redder than MS L2",
            ]

        result = np.select(conditions, labels, default="redder than MS L3")

        df.loc[mask, pos_col] = result

    # --- Below MS ---
    df.loc[y_vals > main_sequence_y[-1], pos_col] = "below the MS"

    return df