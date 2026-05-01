import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .module_xray import (
    load_xray_data,
    parse_coordinates,
    compute_counts_and_errors,
    compute_luminosity,
    classify_sources,
    finalize_xray
)

def x_ray_classification(path_xray: str, BS_RA: float, BS_Decl: float, distance_parsecs: float) -> pd.DataFrame:

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

    print("X-ray classification completed")

    return df






