import numpy as np
from ..utils.colors import get_color_columns

def create_4_regions(
    df,
    graphic_comp,
    main_sequence_y,
    msto_y,
    min_smooth
):
    """
    Split optical sources into 4 regions based on CMD geometry.

    Regions:
    - R1: Below main sequence
    - R2: Between MS and MSTO
    - R3: Right side (RGB / AGB candidates)
    - R4: Left side (HB / BS candidates)
    """

    # --- Copy (safe) ---
    df = df.copy()

    # --- Colors ---
    Color1, Color2 = get_color_columns(graphic_comp)

    x_col = f"{Color1} - {Color2}"
    y_col = f"{Color2}_Mag"
    pos_col = f"position_{graphic_comp}"

    # --- Extract arrays ---
    x = df[x_col].values
    y = df[y_col].values
    pos = df[pos_col].values

    # --- Key MS points ---
    idx_A = np.argmax(main_sequence_y == msto_y)

    x_A = min_smooth[idx_A]
    y_A = msto_y

    x_B = min_smooth[0]
    y_B = main_sequence_y[0]

    y_max_ms = main_sequence_y[-1]

    # --- Line A → B ---
    slope = (x_B - x_A) / (y_B - y_A)
    x_boundary = x_A + slope * (y - y_A)
    #print(msto_y)
    # --- Masks ---
    mask_r1 = y >= y_max_ms

    mask_r2 = (y < y_max_ms) & (y >= y_A)

    mask_upper = (y < y_A) & (y >= y_B)
    mask_lower = y < y_B

    mask_sgb_msto = (pos == "Sub Giant Branch") | (pos == "MSTO")

    mask_r3 = (
        (mask_upper & (x > x_boundary)) |
        (mask_lower & (x > x_B)) |
        mask_sgb_msto
    )

    mask_r4 = (
        ((mask_upper & (x <= x_boundary)) |
         (mask_lower & (x <= x_B))) &
        (~mask_sgb_msto)
    )

    # --- Create regions ---
    region1 = df[mask_r1]
    region2 = df[mask_r2]
    region3 = df[mask_r3]
    region4 = df[mask_r4]

    return region1, region2, region3, region4