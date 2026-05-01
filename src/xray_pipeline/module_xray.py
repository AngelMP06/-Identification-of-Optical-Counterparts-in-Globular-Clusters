import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

import json
import numpy as np
import pandas as pd


# ============================================================
# POSITIONAL UNCERTAINTY
# ============================================================

def radius_95(count: float) -> float:
    """
    Estimate the 95% positional uncertainty radius (arcsec)
    based on total X-ray counts.

    Parameters
    ----------
    count : float
        Total counts detected for the source.

    Returns
    -------
    float
        Positional uncertainty radius in arcseconds.
    """
    if count <= 0 or np.isnan(count):
        return np.nan

    logc = np.log10(count)

    if logc <= 2.1393:
        return 10 ** (-0.4958 * logc + 0.1932)
    elif logc <= 3.3:
        return 10 ** (-0.2064 * logc - 0.4260)

    return np.nan


# ============================================================
# DATA LOADING
# ============================================================

def load_xray_data(path: str) -> pd.DataFrame:
    """
    Load X-ray catalog from JSON file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the JSON file.

    Returns
    -------
    pd.DataFrame
        Raw X-ray catalog.
    """
    with open(path, "r") as f:
        data = json.load(f)
    
    required_cols = ["Fx_0.5-2.5", "Fx_2.5-6.0", "RA", "Decl"]
    df = pd.DataFrame(data)
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


# ============================================================
# COORDINATE TRANSFORMATION
# ============================================================

def parse_coordinates(df: pd.DataFrame, BS_RA: float, BS_Decl: float) -> pd.DataFrame:
    """
    Convert RA/Decl from sexagesimal format to degrees and apply boresight correction.

    Parameters
    ----------
    df : pd.DataFrame
        Input catalog with RA/Decl in string format.
    BS_RA : float
        Boresight correction in RA (degrees).
    BS_Decl : float
        Boresight correction in Declination (degrees).

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with RA and Decl in degrees.
    """
    df = df.copy()

    # RA conversion
    ra_parts = df["RA"].str.extract(r"(\d+)h(\d+)m([\d\.]+)s").astype(float)
    df["RA"] = 15 * (ra_parts[0] + ra_parts[1] / 60 + ra_parts[2] / 3600) + BS_RA

    # Decl conversion
    decl_parts = df["Decl"].str.extract(r"([+-]?\d+)d(\d+)m([\d\.]+)s").astype(float)
    df["Decl"] = decl_parts[0] - decl_parts[1] / 60 - decl_parts[2] / 3600 + BS_Decl

    return df


# ============================================================
# COUNTS & POSITIONAL ERROR
# ============================================================

def compute_counts_and_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract counts from different energy bands and compute total counts
    and positional uncertainty radius.

    Parameters
    ----------
    df : pd.DataFrame
        Input catalog with Xsoft, Xmed, Xhard columns.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with total_count and r95 (degrees).
    """
    df = df.copy()

    xsoft = df["Xsoft"].str.extract(r"(\d+)/([\d\.]+)").astype(float)
    xmed  = df["Xmed"].str.extract(r"(\d+)/([\d\.]+)").astype(float)
    xhard = df["Xhard"].str.extract(r"(\d+)/([\d\.]+)").astype(float)

    df["total_count"] = xsoft[0] + xmed[0] + xhard[0]

    # Convert arcsec → degrees
    df["r95"] = df["total_count"].apply(radius_95) / 3600

    return df


# ============================================================
# PHYSICAL QUANTITIES
# ============================================================

def compute_luminosity(df: pd.DataFrame, distance_parsecs: float) -> pd.DataFrame:
    """
    Compute X-ray luminosities and hardness ratio.

    Parameters
    ----------
    df : pd.DataFrame
        Input catalog with flux columns.
    distance_parsecs : float
        Distance to the source in parsecs.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with luminosities and log-scaled quantities.
    """
    df = df.copy()

    # Convert parsec → cm
    d_cm = distance_parsecs * 3.086e18

    df["Lx_soft"] = df["Fx_0.5-2.5"] * 4 * np.pi * d_cm**2
    df["Lx_hard"] = df["Fx_2.5-6.0"] * 4 * np.pi * d_cm**2
    df["log_LX_soft"] = np.log10(df["Lx_soft"])

    df["Lx_total"] = df["Lx_soft"] + df["Lx_hard"]
    df["HR"] = df["Lx_soft"] / df["Lx_hard"]

    df["log_Lx"] = np.log10(df["Lx_total"])
    df["log_HR"] = np.log10(df["HR"])

    return df


# ============================================================
# SOURCE CLASSIFICATION
# ============================================================

def classify_sources(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify X-ray sources based on luminosity and hardness ratio.

    Parameters
    ----------
    df : pd.DataFrame
        Input catalog with log_Lx and log_HR columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with classification labels.
    """
    df = df.copy()

    conditions = [
        (df["log_Lx"] >= 31.60206) & (-3.996566 * df["log_HR"] + 33.6003433 - df["log_Lx"] < 0),
        (df["log_Lx"] >= 31.60206) & (-3.996566 * df["log_HR"] + 33.6003433 - df["log_Lx"] >= 0),
        (df["log_Lx"] < 31.60206) & (df["log_Lx"] >= 30.60206),
        df["log_Lx"] < 30.60206
    ]

    choices = ["LMXRB", "CV", "CV & AB", "AB"]

    df["class"] = np.select(conditions, choices, default="")

    return df


# ============================================================
# FINAL CLEANING
# ============================================================

def finalize_xray(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove intermediate columns and prepare final catalog.

    Parameters
    ----------
    df : pd.DataFrame
        Input catalog with all computed columns.

    Returns
    -------
    pd.DataFrame
        Cleaned X-ray catalog ready for export or crossmatching.
    """
    df = df.copy()

    df.drop(columns=[
        "Xsoft", "Xmed", "Xhard",
        "Fx_0.5-2.5", "Fx_2.5-6.0",
        "total_count"
    ], inplace=True)

    df.index.name = "id"

    return df