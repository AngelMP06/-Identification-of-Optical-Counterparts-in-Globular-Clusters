import pandas as pd

from src.probability_pipeline.get_probability_counterpart import run_probability_pipeline
from src.optical_pipeline.classify_optical import optical_classification
from src.xray_pipeline.classify_xray import x_ray_classification
from src.find_candidates_pipeline.find_candidates import run_crossmatch_pipeline
from src.load_cluster_data import load_cluster_config

from src.models import FilterParams, MSDetectionParams

def run_full_pipeline(cluster_name: str, 
                      show_optical_plots: bool = False,
                      show_xray_plot: bool = False,
                      show_crossmatch_DaraFrame: bool = False,
                      show_probability_plots: bool = False):

    params = load_cluster_config(cluster_name)

    df_optical = optical_classification(params["path_optical"], params["distance_parsecs"], cluster_name, show_optical_plots)

    df_xray = x_ray_classification(params["path_xray"], params["BS_RA"], params["BS_Decl"], params["distance_parsecs"], cluster_name, show_xray_plot)

    df_matches = run_crossmatch_pipeline(df_optical, df_xray, show_crossmatch_DaraFrame)

    probabilities_candidates = run_probability_pipeline(df_matches, show_probability_plots)
    

    print("\n")
    print("####################################################")
    print("Final probabilities for candidate counterparts:")
    print("####################################################")
    
    print("")
    print("A source is likely to be a counterpart if it has more or equal than 75% on P_counterpart (%)")
    
    if probabilities_candidates.empty:
        print("")
        print("No likely counterparts found")
    else:
        print("")
        print("####################################################")
        print("Likely counterparts found:")
        print("####################################################")
        print("")
        print(probabilities_candidates)
