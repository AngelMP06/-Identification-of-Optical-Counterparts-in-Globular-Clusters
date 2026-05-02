import numpy as np
import pandas as pd

from .module import compute_angular_distance, build_row

def run_crossmatch_pipeline(optical_data: pd.DataFrame, 
                            xray_data: pd.DataFrame, 
                            show_crossmatch_DaraFrame: bool = False, 
                            search_secure_counterparts: bool = False) -> pd.DataFrame:

    """
    Identify optical counterparts for X-ray sources using positional crossmatching.

    - First calculate the angular distance between each X-ray source and all optical sources.
    - Select optical sources within the 95% confidence radius (r95) of the X-ray source.
    - If no sources are found within r95, apply a fallback search using a fixed radius of 2 arcseconds.
    - For each matched candidate, compile a row of data including positional information, photometric measurements, 
    and a simple classification based on Mv and log_Lx.
    - If no candidates are found for an X-ray source, it is skipped.
    - The final output is a DataFrame containing all matched candidates with their associated data.

    Parameters
    ----------
    optical_data : pd.DataFrame
        Optical catalog with positions and photometric classifications.
    xray_data : pd.DataFrame
        X-ray catalog including positions, uncertainties, and classifications.
    show_crossmatch_DaraFrame : bool, optional
        Whether to display the DataFrame of matched candidates (by default False)
    search_secure_counterparts : bool, optional
        If True, the pipeline will search directly within 2 arc seconds (default is False).
    Returns
    -------
    pd.DataFrame
        Table of matched candidates, including positional, photometric,
        and classification information for each association.
    """

    optical_df = optical_data.copy()
    xray_df = xray_data.copy()
    
    # Defining the names of the X-ray sources
    cx_list = xray_df.index.tolist()

    results = []

    # Pre-extract arrays 
    ra = optical_df["RA"].values
    dec = optical_df["Decl"].values

    for CX in cx_list:

        ra0 = xray_df.loc[CX, "RA"]
        dec0 = xray_df.loc[CX, "Decl"]
        r95 = xray_df.loc[CX, "r95"]

        dist = compute_angular_distance(ra, dec, ra0, dec0)

        
        if not search_secure_counterparts:

            # --- FIRST SEARCH: Sources inside r95 degrees ---
            idx = np.where(dist < r95)[0]
            radius_flag = "r95"


            # --- FALLBACK: 2 arcsec ---
            radius = 2 / 3600  # Convert arcsec to degrees
            if len(idx) == 0 and r95 < radius: # type: ignore
                idx = np.where(dist < radius)[0]
                radius_flag = "2"
        else:

            # --- FALLBACK: 2 arcsec ---
            radius = 2 / 3600  # Convert arcsec to degrees
            idx = np.where(dist < radius)[0]
            radius_flag = "2"

        # --- PROCESS MATCHES ---
        n_sources = len(idx)

        if n_sources == 0:
            pass

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
    
    df_matches =  pd.DataFrame(results)
    
    print("\n")
    print("Candidate search completed")

    if show_crossmatch_DaraFrame:
        print("\n")
        print("####################################################")
        print("Possible counterparts found for each X-ray source:")
        print("####################################################")
        print("\n")
        print(df_matches)

    return df_matches