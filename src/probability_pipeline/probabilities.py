import numpy as np

def compute_base_probability(df):

    df["p_counterpart"] = df["AB"] + df["CV"] + df["LMXRB"]
    df["p_not"] = 1 - df["p_counterpart"]

    total_prob = 1 - df.groupby("Xray source")["p_not"].prod()

    return df, total_prob

def apply_weighting(df):

    weights = []

    for _, row in df.iterrows():

        arr = np.array([row["AB"], row["CV"], row["LMXRB"]])
        arr_sorted = np.sort(arr)[::-1]

        if arr_sorted[1] != 0:
            ratio = arr_sorted[0] / arr_sorted[1]
        else:
            ratio = 1e9

        weights.append(row["p_counterpart"] * ratio)

    df["p_weighted"] = weights

    return df

def normalize_probabilities(df, total_prob):

    sum_after = df.groupby("Xray source")["p_weighted"].sum()

    new_probs = []

    for i, row in df.iterrows():

        x_id = row["Xray source"]

        p = row["p_weighted"] * total_prob.loc[x_id] / sum_after.loc[x_id]
        new_probs.append(p)

    df["p_final"] = new_probs

    return df

def apply_radius_penalty(df):

    df.loc[df["radius"] == "2", "p_final"] *= 0.8

    return df

def rescale_classes(df):

    scale = df["p_final"] / df["p_counterpart"]

    df["AB"] *= scale
    df["CV"] *= scale
    df["LMXRB"] *= scale

    return df