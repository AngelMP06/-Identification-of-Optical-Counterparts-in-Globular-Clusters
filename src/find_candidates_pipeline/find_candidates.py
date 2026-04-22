from .module import find_candidates_numpy

def run_crossmatch_pipeline(optical_data, xray_data):

    optical_df = optical_data
    xray_df = xray_data

    df_matches = find_candidates_numpy(optical_df, xray_df, xray_df.index.tolist())
    
    print("Candidate search completed")

    return df_matches