import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from .module_xray import (
    load_xray_data,
    parse_coordinates,
    compute_counts_and_errors,
    compute_luminosity,
    classify_sources,
    finalize_xray,
)

def x_ray_classification(path_xray: str, BS_RA: float, BS_Decl: float, distance_parsecs: float, cluster_name: str, show_plot: bool = False) -> pd.DataFrame:

    """
    Perform X-ray source classification for a globular cluster.

    This function executes the full X-ray analysis pipeline, including
    data loading, coordinate parsing, count and error computation,
    luminosity estimation, and final source classification.

    Parameters
    ----------
    path_xray : str
        Path to the X-ray catalog file.
    BS_RA : float
        Right Ascension of the cluster center.
    BS_Decl : float
        Declination of the cluster center.
    distance_parsecs : float
        Distance to the cluster in parsecs.
    cluster_name : str
        Name of the globular cluster (used for output organization).
    show_plot : bool, optional
        Whether to display the X-ray classification plot (by default False)

    Returns
    -------
    pd.DataFrame
        DataFrame containing the processed and classified X-ray sources.
    """

    df = load_xray_data(path_xray)
    df = parse_coordinates(df, BS_RA, BS_Decl)
    df = compute_counts_and_errors(df)
    df = compute_luminosity(df, distance_parsecs)
    df = classify_sources(df)
    df = finalize_xray(df) 

    if show_plot:
        # --- Define output directory ---
        output_dir = os.path.join("results", cluster_name, "xray_plot")

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # --- Define filename and full path ---
        filename = f"Xray_classification.png"
        full_path = os.path.join(output_dir, filename)
        
        plt.figure(figsize=(10, 8))

        x_axis = np.linspace(-1, 2, 2)
        point_1 = [-0.1, 0.5]
        point_2 = [34, 31.6020599913]

        sns.scatterplot(data=df, x="log_HR", y="log_Lx", hue="class", s = 20)
        for i in range(len(df["log_Lx"])):
            plt.annotate(df.index[i], (df["log_HR"].iloc[i], df["log_Lx"].iloc[i]), fontsize = 8)
        plt.plot(x_axis, [30.6020599913, 30.6020599913], c = "red", label = "Dividing line")
        plt.plot(x_axis, [31.6020599913, 31.6020599913], c = "red")
        plt.plot(point_1, point_2, c = "red")

        plt.legend()

        if not os.path.exists(full_path):
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"New plot saved: {full_path}")

        plt.show()
        
    print("\n")
    print("X-ray classification completed")

    return df






