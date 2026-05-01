import numpy as np
import pandas as pd

from matplotlib.path import Path

from src.models import Points
    
from ..utils.colors import get_color_columns


def AGB_BS_classification(
    df_merged: pd.DataFrame,
    CMD_config: int,
    points: Points
):
    """
    Classify Blue Stragglers (BS) and Asymptotic Giant Branch (AGB) stars.

    - BS: inside polygon C-D-G-H
    - AGB: above the extended line E-F and to the right of E
    """

    df_merged = df_merged.copy()

    # --- Colors ---
    Color1, Color2 = get_color_columns(CMD_config)

    x_col = f"{Color1} - {Color2}"
    y_col = f"{Color2}_Mag"
    pos_col = f"position_{CMD_config}"

    x = np.array(df_merged[x_col].values)
    y = np.array(df_merged[y_col].values)
    pos = np.array(df_merged[pos_col].values)


    # --- Blue Stragglers ---
    polygon = np.array([points.point_C, points.point_D, points.point_G, points.point_H])
    poly_path = Path(polygon)

    points_polygon = np.column_stack((x, y))
    inside = poly_path.contains_points(points_polygon)

    mask_bs = (pos == "Unknown") & inside
    df_merged.loc[mask_bs, pos_col] = "BS"


    # --- AGB ---

    # Line point_E → point_F: y = m*x + b
    m = (points.point_F[1] - points.point_E[1]) / (points.point_F[0] - points.point_E[0])
    b = points.point_E[1] - m * points.point_E[0]

    y_line = m * x + b

    mask_agb = (
        (pos == "Unknown") &
        (x > points.point_E[0]) &
        (y < y_line)
    )

    df_merged.loc[mask_agb, pos_col] = "AGB"

    return df_merged