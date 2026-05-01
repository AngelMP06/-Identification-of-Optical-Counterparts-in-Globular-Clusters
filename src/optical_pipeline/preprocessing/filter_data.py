import numpy as np
import pandas as pd

from src.models import FilterParams

def filter_data(df: pd.DataFrame, filter_params: FilterParams) -> pd.DataFrame:
    """
    Filter and preprocess optical catalog data.

    This function:
    - Computes visible magnitude and absolute magnitude (Mv)
    - Applies quality filters across all photometric bands
    - Removes unnecessary columns

    Parameters
    ----------
    df : pd.DataFrame
        Raw optical catalog data.
    distance_parsecs : float
        Distance to the cluster in parsecs.
    filter_params : FilterParams
        Filtering parameters.

    Returns
    -------
    pd.DataFrame
        Cleaned and filtered dataset ready for CMD analysis.
    """

    # --- Work on a copy (avoid modifying original data) ---
    df = df.copy()


    # --- Apply filtering conditions ---
    bands = ["275", "336", "438", "606", "814"]

    mask = np.ones(len(df), dtype=bool)

    for band in bands:
        mask &= df[f"{band}_Mag"].between(filter_params.Mag_min, filter_params.Mag_max)
        mask &= df[f"{band}_RMS"] < filter_params.RMS_max
        mask &= df[f"{band}_Fit"] < filter_params.Fit_max
        mask &= df[f"{band}_Sharp"].between(filter_params.Sharp_min, filter_params.Sharp_max)

    # Apply cluster membership filter ONCE
    mask &= df["Cluster_Membership"] > filter_params.CM_min

    df_filtered = df[mask].copy()
    df_filtered.reset_index(drop=True, inplace=True)


    # --- Drop unnecessary columns ---
    columns_to_drop = []

    for band in bands:
        columns_to_drop.extend([
            f"{band}_RMS",
            f"{band}_Fit",
            f"{band}_Sharp",
            f"{band}_exp_found",
            f"{band}_exp_well"
        ])

    df_filtered.drop(columns=columns_to_drop, inplace=True)


    # --- Create color indices (CMD features) ---
    df_filtered["275 - 336"] = df_filtered["275_Mag"] - df_filtered["336_Mag"]
    df_filtered["438 - 606"] = df_filtered["438_Mag"] - df_filtered["606_Mag"]
    df_filtered["606 - 814"] = df_filtered["606_Mag"] - df_filtered["814_Mag"]

    return df_filtered