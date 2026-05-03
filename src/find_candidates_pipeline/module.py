import numpy as np
import pandas as pd


def compute_angular_distance(ra, dec, ra0, dec0):
    """
    Calculate the angular distance of the optical source and the xray source (degrees)
    """
    return np.sqrt(
        ((ra - ra0) * np.cos(np.deg2rad(dec)))**2 +
        (dec - dec0)**2
    )

def classify_optical_xray(Mv, log_Lx):

    """
    Simple classification based on Mv and log_Lx, 34 and 36.2 are empirically derived thresholds. Those values can be found onthe next paper:
    https://www.cambridge.org/core/journals/proceedings-of-the-international-astronomical-union/article/observational-evidence-for-the-origin-of-xray-sources-in-globular-clusters/23A31D04F0D308EC4B1CC2727290ECE8
    """

    z = 0.4 * Mv + log_Lx
     
    if z < 34:
        return "AB"
    elif z < 36.2:
        return "CV"
    else:
        return "LMXRB"
    

def build_row(CX, num, i_opt, optical_df, xray_df, radius_flag, n_sources):

    """
    Build a row for the matched candidate, including classification and photometric data.
    """

    Mv = optical_df.loc[i_opt, "Mv"]
    log_Lx = xray_df.loc[CX, "log_LX_soft"]

    classification = classify_optical_xray(Mv, log_Lx)

    return {
        "id_candidate": f"{CX}{chr(97 + num)}",
        "Xray source": CX,
        "opt_id": optical_df.loc[i_opt, "Id"],
        "Hardness_classification": xray_df.loc[CX, "class"],
        "Mv vs xray": classification,

        # Photometry
        "275 - 336": optical_df.loc[i_opt, "275 - 336"],
        "336_Mag": optical_df.loc[i_opt, "336_Mag"],
        "438 - 606": optical_df.loc[i_opt, "438 - 606"],
        "606_Mag": optical_df.loc[i_opt, "606_Mag"],
        "606 - 814": optical_df.loc[i_opt, "606 - 814"],
        "814_Mag": optical_df.loc[i_opt, "814_Mag"],
        "pos_0": optical_df.loc[i_opt, "position_0"],
        "pos_1": optical_df.loc[i_opt, "position_1"],
        "pos_2": optical_df.loc[i_opt, "position_2"],

        # Matching info
        "radius": radius_flag,
        "n_sources": n_sources
    }