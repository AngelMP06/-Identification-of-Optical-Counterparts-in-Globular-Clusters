def finalize_output(df):

    prob_sum = df.groupby("Xray source")["p_final"].sum()

    df["total_prob"] = df["Xray source"].map(prob_sum)

    df["Prob_AB (%)"] = (df["AB"] * 100).round(1)
    df["Prob_CV (%)"] = (df["CV"] * 100).round(1)
    df["Prob_LMXRB (%)"] = (df["LMXRB"] * 100).round(1)
    df["P_counterpart (%)"] = (df["p_final"] * 100).round(1)
    df["P_Xray_source (%)"] = (df["total_prob"] * 100).round(1)

    return df[[
        "id",
        "Prob_AB (%)",
        "Prob_CV (%)",
        "Prob_LMXRB (%)",
        "P_counterpart (%)",
        "P_Xray_source (%)"
    ]]