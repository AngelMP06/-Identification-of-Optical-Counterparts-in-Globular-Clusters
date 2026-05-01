import pandas as pd

from .scores import initialize_scores, apply_hardness_score, apply_mv_xray_score, apply_optical_score
from .probabilities import compute_base_probability, apply_weighting, normalize_probabilities, apply_radius_penalty, rescale_classes
from .final_output import finalize_output

def run_probability_pipeline(df_matches: pd.DataFrame, show_probability_plots: bool = False) -> pd.DataFrame:

    """
    Compute the probability that each optical source is the true counterpart
    of an X-ray source.

    This function applies a scoring and probability framework combining
    X-ray properties, optical information, and positional constraints.
    Scores are converted into probabilities, weighted, normalized, and
    adjusted with penalties to produce a final likelihood for each candidate.

    Pipeline steps:
    1) Initialize the scores of being AB, CV, LMXRB
    2) Apply score based on the hardness classification, Mv vs X-ray, and optical classification
    3) Apply weighting and normalization
    4) Penalize candidates based on matching radius
    5) Rescale probabilities across classes
    6) Generate the final output table

    Parameters
    ----------
    df_matches : pd.DataFrame
        DataFrame containing matched optical-X_ray candidates.
    show_probability_plots : bool, optional
        Whether to display the results of the probability pipeline, by default False
    Returns
    -------
    pd.DataFrame
        DataFrame with final probabilities and classification scores
        for each candidate counterpart.
    """

    df = df_matches.copy()

    # --- SCORES ---
    df = initialize_scores(df)
    df = apply_hardness_score(df)
    df = apply_mv_xray_score(df)
    df = apply_optical_score(df)

    # --- GET PROBABILITIES ---
    df, total_prob = compute_base_probability(df)
    df = apply_weighting(df)
    df = normalize_probabilities(df, total_prob)
    df = apply_radius_penalty(df)
    df = rescale_classes(df)

    # --- FINAL ---
    df_final = finalize_output(df)      
    print("\n")
    print("Probability pipeline completed successfully.")

    if show_probability_plots:
        print("\n")
        print("####################################################")
        print("Probability pipeline results:")
        print("####################################################")
        print("\n")
        print(df_final)
    filter = df_final["P_counterpart (%)"] >= 80

    return df_final.loc[filter]