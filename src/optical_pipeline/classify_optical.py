import pandas as pd

from optical_pipeline.classification_sources.classification import classify_optical_data
from optical_pipeline.preprocessing.load_data import load_optical_data
from optical_pipeline.preprocessing.filter_data import filter_data
from optical_pipeline.main_sequence_detection.find_main_sequence import find_main_sequence

from models import FilterParams, MSDetectionParams, MainSequenceResults, Points

def optical_classification(path_optical : str, 
                           distance_parsecs : float, 
                           bins_division : int = 20, 
                           filter_params: FilterParams | None = None,
                           ms_params: MSDetectionParams | None = None) -> pd.DataFrame:
    """
    Perform optical source classification using Color-Magnitude Diagrams (CMDs).

    This function runs the full optical analysis pipeline, including data loading,
    quality filtering, main sequence detection, and classification of sources
    based on their position in the CMD.

    The classification is performed across multiple CMD configurations
    (CMD_config), allowing a more robust identification of stellar populations.

    Parameters
    ----------
    path_optical : str
        Path to the optical catalog file.
    distance_parsecs : float
        Distance to the cluster in parsecs.
    bins_division : int, optional
        Resolution parameter for histogram-based methods (default is 20).
    filter : FilterParams, optional
        Filtering parameters for data quality (default is a FilterParams instance with default values).
    ms_detection : MSDetectionParams, optional
        Parameters for main sequence detection (default is an MSDetectionParams instance with default values).
        
    default parameters for filtering and classification can be adjusted as needed.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the filtered and classified optical sources,
        including classification labels for each CMD configuration.
    """

    # Set default parameters if not provided
    
    if filter_params is None:
        filter_params = FilterParams()

    if ms_params is None:
        ms_params = MSDetectionParams()
    

    # --- Step 1: We load all the optical data we need and assign the column names ---
    optical_data = load_optical_data(path_optical, distance_parsecs)


    # --- Step 2: Filter data based on criteria ---
    df = filter_data(optical_data, filter_params)


    for CMD_config in range(3):
        
        df[f"position_{CMD_config}"] = "Unknown"

        # --- Step 3: We now are going to find the left and right edge of the main sequence and main sequence turn off (MSTO) ---
        main_sequence_results, points = find_main_sequence(df, bins_division, CMD_config, ms_params)
        

        # --- Step 4: Classify stars based on their position in the CMD relative to the main sequence and Giant Branch ---
        df_classified = classify_optical_data(df, CMD_config, bins_division, main_sequence_results, points)


        column = f"position_{CMD_config}"
        df[column] = df_classified[column]

    print("Optical classification completed")

    return df


