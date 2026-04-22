def finalize_output(df):

    prob_sum = df.groupby("Xray source")["p_final"].sum()

    df["total_prob"] = df["Xray source"].map(prob_sum)

    df["AB(%)"] = (df["AB"] * 100).round(1)
    df["CV(%)"] = (df["CV"] * 100).round(1)
    df["LMXRB(%)"] = (df["LMXRB"] * 100).round(1)
    df["p_final(%)"] = (df["p_final"] * 100).round(1)
    df["total_prob(%)"] = (df["total_prob"] * 100).round(1)

    return df[[
        "id",
        "AB(%)",
        "CV(%)",
        "LMXRB(%)",
        "p_final(%)",
        "total_prob(%)"
    ]]