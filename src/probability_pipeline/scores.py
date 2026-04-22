def initialize_scores(df):
    df = df.copy()
    df["AB"] = 0.0
    df["CV"] = 0.0
    df["LMXRB"] = 0.0
    return df

def apply_hardness_score(df, HC_score):
    for i, row in df.iterrows():

        HC = row["Hardness_classification"]

        if HC == "LMXRB":
            df.loc[i, "LMXRB"] += HC_score

        elif HC == "CV":
            df.loc[i, "CV"] += HC_score

        elif HC == "CV & AB":
            df.loc[i, "CV"] += HC_score / 2
            df.loc[i, "AB"] += HC_score / 2

        elif HC == "AB":
            df.loc[i, "AB"] += HC_score

    return df

def apply_mv_xray_score(df, MV_XR_score):
    for i, row in df.iterrows():

        MV_XR = row["Mv vs xray"]

        if MV_XR == "LMXRB":
            df.loc[i, "LMXRB"] += MV_XR_score

        elif MV_XR == "CV":
            df.loc[i, "CV"] += MV_XR_score

        elif MV_XR == "AB":
            df.loc[i, "AB"] += MV_XR_score

    return df

def apply_optical_score(df, optical_score):

    for i, row in df.iterrows():

        for j in range(3):

            pos = row[f"pos_{j}"]

            # --- BLUE SIDE ---
            if pos == "bluer than MS L1":

                if row["LMXRB"] > row["CV"] and row["LMXRB"] > row["AB"]:
                    df.loc[i, "LMXRB"] += (optical_score - (optical_score/6)*(j-1)) / 3

                elif row["CV"] > row["AB"]:
                    df.loc[i, "CV"] += (optical_score - (optical_score/6)*(j-1)) / 3

                else:
                    df.loc[i, "LMXRB"] += optical_score / 6
                    df.loc[i, "CV"] += optical_score / 6

            # --- RED SIDE ---
            elif pos in ["redder than MS L1", "sub-sub-giant branch", "Red giant branch"]:

                if row["AB"] > row["CV"] and row["AB"] > row["LMXRB"]:
                    df.loc[i, "AB"] += (optical_score + (optical_score/6)*(j-1)) / 3
                else:
                    df.loc[i, "AB"] += (optical_score + (optical_score/6)*(j-1)) / 5

            # --- MAIN SEQUENCE ---
            elif pos in ["MS", "MSTO"]:

                if row["AB"] > row["CV"] and row["AB"] > row["LMXRB"]:
                    df.loc[i, "AB"] += optical_score / 6

    return df