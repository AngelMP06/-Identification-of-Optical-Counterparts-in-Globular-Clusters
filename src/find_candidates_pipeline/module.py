import numpy as np
import pandas as pd

def find_candidates_numpy(optical_df, xray_df, cx_list):
    """
    Find optical candidates using:
    - first: 0.6 arcsec
    - fallback: 2 arcsec (if none found)
    """

    results = []

    # Pre-extract arrays (faster)
    ra = optical_df["RA"].values
    dec = optical_df["Decl"].values

    for CX in cx_list:

        ra0 = xray_df.loc[CX, "RA"]
        dec0 = xray_df.loc[CX, "Decl"]
        r95 = xray_df.loc[CX, "r95"]

        dist = compute_angular_distance(ra, dec, ra0, dec0)

        # --- FIRST SEARCH: Sources inside r95 degrees ---
        idx = np.where(dist < r95)[0]
        radius_flag = "r95"

        # --- FALLBACK: 2 arcsec ---
        radius = 2 / 3600  # Convert arcsec to degrees
        if len(idx) == 0 and r95 < radius:
            idx = np.where(dist < radius)[0]
            radius_flag = "2"

        # --- PROCESS MATCHES ---
        n_sources = len(idx)

        if n_sources == 0:
            # Keep empty case
            results.append({
                "id": CX,
                "opt_id": None,
                "radius": radius_flag,
                "n_sources": 0
            })
            continue

        for num, i_opt in enumerate(idx):

            row = build_row(
                CX,
                num,
                i_opt,
                optical_df,
                xray_df,
                radius_flag,
                n_sources
            )

            results.append(row)

    return pd.DataFrame(results)

def compute_angular_distance(ra, dec, ra0, dec0):
    """
    Small-angle approximation for angular distance (degrees)
    """
    return np.sqrt(
        ((ra - ra0) * np.cos(np.deg2rad(dec)))**2 +
        (dec - dec0)**2
    )

def classify_optical_xray(Mv, log_Lx):
    z = 0.4 * Mv + log_Lx

    if z < 34:
        return "AB"
    elif z < 36.2:
        return "CV"
    else:
        return "LMXRB"
    

def build_row(CX, num, i_opt, optical_df, xray_df, radius_flag, n_sources):

    Mv = optical_df.loc[i_opt, "Mv"]
    log_Lx = xray_df.loc[CX, "log_LX_soft"]

    classification = classify_optical_xray(Mv, log_Lx)

    return {
        "id": f"{CX}_{num+1}",
        "Xray source": CX,
        "opt_id": optical_df.loc[i_opt, "Id"],
        "Hardness_classification": xray_df.loc[CX, "class"],
        "Mv vs xray": classification,

        # Photometry
        "pos_0": optical_df.loc[i_opt, "position_0"],
        "275 - 336": optical_df.loc[i_opt, "275 - 336"],
        "336_Mag": optical_df.loc[i_opt, "336_Mag"],

        "pos_1": optical_df.loc[i_opt, "position_1"],
        "438 - 606": optical_df.loc[i_opt, "438 - 606"],
        "606_Mag": optical_df.loc[i_opt, "606_Mag"],

        "pos_2": optical_df.loc[i_opt, "position_2"],
        "606 - 814": optical_df.loc[i_opt, "606 - 814"],
        "814_Mag": optical_df.loc[i_opt, "814_Mag"],

        # Matching info
        "radius": radius_flag,
        "n_sources": n_sources
    }