import pandas as pd
import math

COLUMNS = [
    "X", "Y",
    "275_Mag", "275_RMS", "275_Fit", "275_Sharp", "275_exp_found", "275_exp_well",
    "336_Mag", "336_RMS", "336_Fit", "336_Sharp", "336_exp_found", "336_exp_well",
    "438_Mag", "438_RMS", "438_Fit", "438_Sharp", "438_exp_found", "438_exp_well",
    "606_Mag", "606_RMS", "606_Fit", "606_Sharp", "606_exp_found", "606_exp_well",
    "814_Mag", "814_RMS", "814_Fit", "814_Sharp", "814_exp_found", "814_exp_well",
    "Cluster_Membership",
    "RA", "Decl",
    "Id",
    "Iteration_found"
]


def load_optical_data(path: str, distance_parsecs: float) -> pd.DataFrame:
    """
    Load optical catalog from a whitespace-separated file.

    Parameters
    ----------
    path : str
        Path to the input file.
    distance_parsecs : float
        Distance to the cluster in parsecs.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names.
    """

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None
    )

    # --- If the columns are not as expected, raise an error ---
    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Expected {len(COLUMNS)} columns, got {df.shape[1]}"
        )

    df.columns = COLUMNS


    # --- Compute visible magnitude and absolute magnitude ---
    df["Visible"] = (df["438_Mag"] + df["606_Mag"]) / 2
    df["Mv"] = df["Visible"] - 5 * math.log10(distance_parsecs / 10)


    # --- Drop unnecessary columns ---
    columns_to_drop = ["X", "Y", "Iteration_found"]

    df.drop(columns=columns_to_drop, inplace=True)

    return df