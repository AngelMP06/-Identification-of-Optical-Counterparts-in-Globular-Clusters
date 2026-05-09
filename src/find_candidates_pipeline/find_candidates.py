import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text

import os

from src.optical_pipeline.utils.colors import get_color_columns
from .module import compute_angular_distance, build_row

def run_crossmatch_pipeline(optical_data: pd.DataFrame, 
                            xray_data: pd.DataFrame, 
                            cluster_name: str,
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
    cluster_name : str
        Name of the star cluster for result organization.
    show_crossmatch_DaraFrame : bool, optional
        Whether to display the CMD plots with all possible counterpart (default is False).
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

        for CMD_config in range(3):

            Color1, Color2 = get_color_columns(CMD_config)

            # --- Define output directory ---
            output_dir = os.path.join("results", cluster_name, "candidates")

            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # --- Define filename and full path ---
            filename = f"CMD_{Color1}_{Color2}_counterparts.png"
            full_path = os.path.join(output_dir, filename)

            plt.figure(figsize=(10, 8))

            # Store text objects
            texts = []

            # Background stars
            sns.scatterplot(
                data=optical_data,
                x=f"{Color1} - {Color2}",
                y=f"{Color2}_Mag",
                s=2,
                c="gray"
            )

            # Candidate points
            sns.scatterplot(
                data=df_matches,
                x=f"{Color1} - {Color2}",
                y=f"{Color2}_Mag",
                s=80,
                hue=f"pos_{CMD_config}",
                edgecolor="black",
                linewidth=0.5
            )

            # Add labels (store them)
            for _, row in df_matches.iterrows():
                texts.append(
                    plt.text(
                        row[f"{Color1} - {Color2}"],
                        row[f"{Color2}_Mag"],
                        row["id_candidate"],
                        fontsize=9,
                        weight="bold",
                        bbox=dict(
                            facecolor='white',
                            alpha=0.7,
                            edgecolor='none',
                            pad=1
                        )
                    )
                )

            # Apply adjustText to avoid overlaps
            adjust_text(
                texts,
                expand=(1.2, 1.2),
                force_text=(0.5, 0.5),
                arrowprops=dict(
                    arrowstyle="-",
                    color="gray",
                    lw=0.5
                )
            )

            plt.gca().invert_yaxis()
            plt.title("Color-Magnitude Diagram (F275W - F336W)")
            plt.xlabel(f"{Color1} - {Color2}")
            plt.ylabel(f"{Color2}_Mag")
            plt.legend(title="CMD Region")

            # Save logic: check if it exists before saving
            if not os.path.exists(full_path):
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
                print(f"New plot saved: {full_path}")

            plt.show()

    df_matches.drop(columns=["275 - 336", "336_Mag", "438 - 606", "606_Mag", "606 - 814", "814_Mag"], inplace=True)

    return df_matches