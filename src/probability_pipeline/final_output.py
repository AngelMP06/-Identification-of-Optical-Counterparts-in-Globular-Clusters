def finalize_output(df):

    """
    Format and summarize the final probability results for each candidate.

    This function aggregates probabilities per X-ray source and converts
    all relevant quantities into percentage form for easier interpretation.
    It prepares a clean output table containing the final class probabilities,
    the counterpart likelihood, and the total probability associated with
    each X-ray source.
     """

    prob_sum = df.groupby("Xray source")["p_final"].sum()

    df["total_prob"] = df["Xray source"].map(prob_sum)

    df["Prob_AB (%)"] = (df["AB"] * 100).round(1)
    df["Prob_CV (%)"] = (df["CV"] * 100).round(1)
    df["Prob_LMXRB (%)"] = (df["LMXRB"] * 100).round(1)
    df["P_counterpart (%)"] = (df["p_final"] * 100).round(1)

    return df[["id_candidate", "Xray source", "opt_id", "Hardness_classification", "Mv vs xray", 
               "pos_0", "pos_1", "pos_2", "radius", "n_sources", 
               "Prob_AB (%)", "Prob_CV (%)", "Prob_LMXRB (%)", "P_counterpart (%)"]]