import numpy as np
from matplotlib.path import Path

from ..utils.colors import get_color_columns


def AGB_BS_classification(
    df,
    graphic_comp,
    C, D, E, F, G, H
):
    """
    Classify Blue Stragglers (BS) and AGB stars.

    - BS: inside polygon C-D-G-H
    - AGB: above line E-F and to the right of E
    """

    df = df.copy()

    # --- Colors ---
    Color1, Color2 = get_color_columns(graphic_comp)

    x_col = f"{Color1} - {Color2}"
    y_col = f"{Color2}_Mag"
    pos_col = f"position_{graphic_comp}"

    x = df[x_col].values
    y = df[y_col].values
    pos = df[pos_col].values

    # =========================
    # --- Blue Stragglers ---
    # =========================

    polygon = np.array([C, D, G, H])
    poly_path = Path(polygon)

    points = np.column_stack((x, y))
    inside = poly_path.contains_points(points)

    mask_bs = (pos == "Unknown") & inside
    df.loc[mask_bs, pos_col] = "BS"

    # =========================
    # --- AGB ---
    # =========================

    # Line E → F: y = m*x + b
    m = (F[1] - E[1]) / (F[0] - E[0])
    b = E[1] - m * E[0]

    y_line = m * x + b

    mask_agb = (
        (pos == "Unknown") &
        (x > E[0]) &
        (y < y_line)
    )

    df.loc[mask_agb, pos_col] = "AGB"

    return df