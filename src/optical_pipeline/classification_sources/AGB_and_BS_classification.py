import numpy as np
from matplotlib.path import Path

from ..utils.colors import get_color_columns


def AGB_BS_classification(
    df_merged,
    graphic_comp,
    point_C, point_D, point_E, point_F, point_G, point_H
):
    """
    Classify Blue Stragglers (BS) and Asymptotic Giant Branch (AGB) stars.

    - BS: inside polygon C-D-G-H
    - AGB: above line E-F and to the right of E
    """

    df_merged = df_merged.copy()

    # --- Colors ---
    Color1, Color2 = get_color_columns(graphic_comp)

    x_col = f"{Color1} - {Color2}"
    y_col = f"{Color2}_Mag"
    pos_col = f"position_{graphic_comp}"

    x = df_merged[x_col].values
    y = df_merged[y_col].values
    pos = df_merged[pos_col].values

    # =========================
    # --- Blue Stragglers ---
    # =========================

    polygon = np.array([point_C, point_D, point_G, point_H])
    poly_path = Path(polygon)

    points = np.column_stack((x, y))
    inside = poly_path.contains_points(points)

    mask_bs = (pos == "Unknown") & inside
    df_merged.loc[mask_bs, pos_col] = "BS"

    # =========================
    # --- AGB ---
    # =========================

    # Line point_E → point_F: y = m*x + b
    m = (point_F[1] - point_E[1]) / (point_F[0] - point_E[0])
    b = point_E[1] - m * point_E[0]

    y_line = m * x + b

    mask_agb = (
        (pos == "Unknown") &
        (x > point_E[0]) &
        (y < y_line)
    )

    df_merged.loc[mask_agb, pos_col] = "AGB"

    return df_merged