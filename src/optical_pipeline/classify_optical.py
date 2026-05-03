import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from src.optical_pipeline.classification_sources.classification import classify_optical_data
from src.optical_pipeline.preprocessing.load_data import load_optical_data
from src.optical_pipeline.preprocessing.filter_data import filter_data
from src.optical_pipeline.main_sequence_detection.find_main_sequence import find_main_sequence
from src.optical_pipeline.utils.colors import get_color_columns

from src.models import FilterParams, MSDetectionParams, MainSequenceResults, Points

def optical_classification(path_optical : str, 
                           distance_parsecs : float,
                           cluster_name : str,
                           show_plots : bool = False,
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

    Filter and MSDecetion Parameters can be adjusted to optimize the classification for different clusters or datasets.

    Parameters
    ----------
    path_optical : str
        Path to the optical catalog file.
    distance_parsecs : float
        Distance to the cluster in parsecs.
    cluster_name : str
        Name of the cluster being analyzed (used for output organization).
    show_plots : bool, optional
        Whether to display the CMD plots with classifications (default is False).
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


        # --- Optional: Plot the classified CMD for this configuration ---
        if show_plots:

            Color1, Color2 = get_color_columns(CMD_config)
            
            # --- Define output directory ---
            output_dir = os.path.join("results", cluster_name, "optical_plots")

            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # --- Define filename and full path ---
            filename = f"CMD_{Color1}_{Color2}_classification.png"
            full_path = os.path.join(output_dir, filename)
            
            plt.figure(figsize=(10, 8))
            
            color_map = {
                # Main branches
                "RSGB": "#FF0000",                # Red
                "RGB": "#FF4500",                 # Less red to orange (OrangeRed)
                "SGB": "#FF8C00",                 # Almost orange (DarkOrange)
                "MSTO": "#FFA500",                # Orange
                "MS": "#FEFE00",                  # Yellow
                "HB": "#0000FF",                  # Blue
                "AGB": "#BF00FF",                 # Between red and blue (Purple/Violet)
                "BS": "#00FFFF",                  # Between blue and orange (Cyan)
                
                # Specials
                "S-SGB": "#FF7C67",               # Redder than orange (Light Salmon)
                "redder than MS L1": "#FF2323",   # Redder than yellow (Red)
                "bluer than MS L1": "#9787FB",    # Bluer than yellow (Medium Slate Blue)
                "bluer than MSTO L1": "#7462EC",  # Bluer than orange (Slate Blue)
                "bluer than SGB L1": "#9D92EA",   # Bluer than reddish orange (Medium Purple)
                "bluer than MS L2": "#03227E",    # Blue (Dark Blue)
                "redder than MS L2": "#920000",   # Red (Dark Red)
                "RS": "#FF9B9B",                  # Red Straggler (Light Coral)
                
                # Discarded
                "Unknown": "#808080",              # Gray
                "below the MS": "#8B4513" # Brown (Saddle Brown)
                }

            

            sns.scatterplot(data=df_classified, 
                            x=f"{Color1} - {Color2}", 
                            y=f"{Color2}_Mag", 
                            hue=column, 
                            legend = True, 
                            s=20,
                            alpha=0.7,
                            palette=color_map)
            
            plt.gca().invert_yaxis()
            plt.title(f"CMD {Color1}_Mag - {Color2}_Mag")
            plt.xlabel(f"{Color1} - {Color2}")
            plt.ylabel(f"{Color2}_Mag")

            # Save logic: check if it exists before saving
            if not os.path.exists(full_path):
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
                print(f"New plot saved: {full_path}")
    
            plt.show()

    print("\n")
    print("Optical classification completed")

    return df


