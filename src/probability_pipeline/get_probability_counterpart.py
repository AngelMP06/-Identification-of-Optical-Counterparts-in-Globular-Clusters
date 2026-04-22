from .scores import initialize_scores, apply_hardness_score, apply_mv_xray_score, apply_optical_score
from .probabilities import compute_base_probability, apply_weighting, normalize_probabilities, apply_radius_penalty, rescale_classes
from .final_output import finalize_output

def run_probability_pipeline(df_matches):

    df = df_matches.copy()
    
    # --- CLEAN ---
    df = df[(df["opt_id"].notna()) & (df["n_sources"] > 0)].reset_index(drop=True)
    
    # --- SCORES ---
    df = initialize_scores(df)
    df = apply_hardness_score(df, HC_score=0.25)
    df = apply_mv_xray_score(df, MV_XR_score=0.25)
    df = apply_optical_score(df, optical_score=0.5)

    # --- GET PROBABILITIES --- 
    df, total_prob = compute_base_probability(df)
    df = apply_weighting(df)
    df = normalize_probabilities(df, total_prob)
    df = apply_radius_penalty(df)
    df = rescale_classes(df)

    # --- FINAL ---
    df_final = finalize_output(df)

    return df_final