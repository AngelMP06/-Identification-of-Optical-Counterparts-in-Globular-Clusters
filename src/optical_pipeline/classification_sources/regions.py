import numpy as np
import pandas as pd

from src.models import MainSequenceResults, Points

from ..utils.colors import get_color_columns

def create_4_regions(
    df: pd.DataFrame,
    CMD_config: int,
    main_sequence_results: MainSequenceResults,
    points: Points
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Points]:
    """
    Split optical sources into 4 regions based on CMD geometry.

    Regions:
    - R1: Below main sequence
    - R2: Between MS and MSTO
    - R3: Right side (RGB / AGB candidates)
    - R4: Left side (HB / BS candidates)

    R3 and R4 are separated by a line formed between the top left corner of the MS (point A) 
    and the top left corner of the RGB (point B).
    """

    # --- Copy (safe) ---
    df = df.copy()


    # --- Colors ---
    Color1, Color2 = get_color_columns(CMD_config)

    x_col = f"{Color1} - {Color2}"
    y_col = f"{Color2}_Mag"
    pos_col = f"position_{CMD_config}"


    # --- Extract arrays ---
    x = df[x_col].values
    y = df[y_col].values
    pos = df[pos_col].values


    # --- Key MS points ---
    idx_A = np.argmax(main_sequence_results.main_sequence_y == main_sequence_results.msto_y)

    point_A = np.zeros(2)
    point_B = np.zeros(2)

    point_A[0] = main_sequence_results.min_smooth[idx_A]
    point_A[1] = main_sequence_results.msto_y

    point_B[0] = main_sequence_results.min_smooth[0]
    point_B[1] = main_sequence_results.main_sequence_y[0]

    y_max_ms = main_sequence_results.main_sequence_y[-1]


    # --- Line A → B ---
    slope = (point_B[0] - point_A[0]) / (point_B[1] - point_A[1])
    x_boundary = point_A[0] + slope * (y - point_A[1])


    # --- Masks ---
    mask_r1 = y >= y_max_ms

    mask_r2 = (y < y_max_ms) & (y >= point_A[1])

    mask_upper = (y < point_A[1]) & (y >= point_B[1])
    mask_lower = y < point_B[1]

    mask_sgb_msto = (pos == "Sub Giant Branch") | (pos == "MSTO")

    mask_r3 = (
        (mask_upper & (x > x_boundary)) |
        (mask_lower & (x > point_B[0])) |
        mask_sgb_msto
    )

    mask_r4 = (
        ((mask_upper & (x <= x_boundary)) |
         (mask_lower & (x <= point_B[0]))) &
        (~mask_sgb_msto)
    )


    # --- Create regions ---
    region1 = df[mask_r1]
    region2 = df[mask_r2]
    region3 = df[mask_r3]
    region4 = df[mask_r4]
    
    points.point_A = point_A
    points.point_B = point_B

    return region1, region2, region3, region4, points