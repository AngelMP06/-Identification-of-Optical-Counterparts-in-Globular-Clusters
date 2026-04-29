from src.probability_pipeline.get_probability_counterpart import run_probability_pipeline
from src.optical_pipeline.classify_optical import optical_classification
from src.xray_pipeline.classify_xray import x_ray_classification
from src.find_candidates_pipeline.find_candidates import run_crossmatch_pipeline
from src.load_cluster_data import load_cluster_config

params = load_cluster_config("NGC_6809")


df_optical = optical_classification(distance_parsecs = params["distance_parsecs"], 
                      path_optical = params["path_optical"])

df_xray = x_ray_classification(
    path_xray=params["path_xray"],
    BS_RA = params["BS_RA"],
    BS_Decl = params["BS_Decl"],
    distance_parsecs=params["distance_parsecs"]
)

df_matches = run_crossmatch_pipeline(df_optical, df_xray)

probabilities_candidates = run_probability_pipeline(df_matches)

print(probabilities_candidates)

print("A source is likely to be a counterpart if it has more than 80% on P_counterpart (%)")