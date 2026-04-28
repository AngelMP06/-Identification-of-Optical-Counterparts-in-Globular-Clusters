from pathlib import Path
import json

def load_cluster_config(cluster_name):

    """
    Load configuration parameters for a given globular cluster.

    This function reads a JSON configuration file located in the cluster's
    data directory and returns the relevant parameters needed for the pipeline.

    The configuration file must include:
    - Paths to X-ray and optical data files
    - Boresight corrections (BS_RA, BS_Decl)
    - Distance to the cluster in parsecs

    Parameters
    ----------
    cluster_name : str
        Name of the cluster (must match a folder in /data).

    Returns
    -------
    dict
        Dictionary containing:
        - path_xray : Path to the X-ray data file
        - path_optical : Path to the optical data file
        - BS_RA : float
        - BS_Decl : float
        - distance_parsecs : float
    """

    base_path = Path(__file__).resolve().parent.parent
    cluster_path = base_path / "data" / cluster_name

    with open(cluster_path / "config.json", "r") as f:
        config = json.load(f)

    return {
        "path_xray": cluster_path / config["xray_file"],
        "path_optical": cluster_path / config["optical_file"],
        "BS_RA": config["BS_RA"],
        "BS_Decl": config["BS_Decl"],
        "distance_parsecs": config["distance_parsecs"]
    }