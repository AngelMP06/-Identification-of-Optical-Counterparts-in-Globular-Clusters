def initialize_scores(df):

    """
    Initialize the classification score columns for each source.

    This function creates score containers for the main classes (AB, CV, LMXRB),
    setting all initial values to zero before applying scoring rules.
    """

    df = df.copy()
    df["AB"] = 0.0
    df["CV"] = 0.0
    df["LMXRB"] = 0.0
    return df

def apply_hardness_score(df, HC_score = 0.25):

    """
    Assign scores based on X-ray hardness classification.

    Each source receives a contribution to its class scores depending on its
    hardness-based classification.
    """

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

def apply_mv_xray_score(df, MV_XR_score = 0.25):

    """
    Assign scores based on the Mv-Lx (optical magnitude vs X-ray luminosity) relation.

    This step updates class scores using the classification derived from the
    Mv vs X-ray relation.
    """

    for i, row in df.iterrows():

        MV_XR = row["Mv vs xray"]

        if MV_XR == "LMXRB":
            df.loc[i, "LMXRB"] += MV_XR_score

        elif MV_XR == "CV":
            df.loc[i, "CV"] += MV_XR_score

        elif MV_XR == "AB":
            df.loc[i, "AB"] += MV_XR_score

    return df

def apply_optical_score(df, optical_score = 0.5):

    """
    This function evaluates the location of each source relative to the Main
    Sequence across different CMD configurations and updates class scores
    accordingly. The scoring strategy is designed to reinforce the most
    physically consistent classification while avoiding ambiguity so we 
    get a secure classification.

    Scoring rules:
    1) Blue side:
    Sources bluer than the Main Sequence are more likely CV or LMXRB.
    The score is assigned to the most probable class based on previous
    (X-ray and Mv-Lx) scores.

    2) Red side:
    Sources redder than the Main Sequence are more likely AB.
    A stronger boost is applied if AB is already the dominant class;
    otherwise, a smaller contribution is assigned.

    3) Main Sequence:
    Sources on the MS or MSTO slightly favor AB classification,
    but only if AB is already the most probable class, to avoid
    artificially inflating ambiguous cases.
    """

    for i, row in df.iterrows():

        for graphic_comp in range(3):

            pos = row[f"pos_{graphic_comp}"]

            # --- BLUE SIDE ---
            if pos == "bluer than MS L1":

                if row["LMXRB"] > row["CV"] and row["LMXRB"] > row["AB"]:
                    df.loc[i, "LMXRB"] += (optical_score - (optical_score/6)*(graphic_comp-1)) / 3

                elif row["CV"] > row["AB"]:
                    df.loc[i, "CV"] += (optical_score - (optical_score/6)*(graphic_comp-1)) / 3

                else:
                    df.loc[i, "LMXRB"] += optical_score / 6
                    df.loc[i, "CV"] += optical_score / 6

            # --- RED SIDE ---
            elif pos in ["redder than MS L1", "sub-sub-giant branch", "Red giant branch"]:

                if row["AB"] > row["CV"] and row["AB"] > row["LMXRB"]:
                    df.loc[i, "AB"] += (optical_score + (optical_score/6)*(graphic_comp-1)) / 3
                else:
                    df.loc[i, "AB"] += (optical_score + (optical_score/6)*(graphic_comp-1)) / 5

            # --- MAIN SEQUENCE ---
            elif pos in ["MS", "MSTO"]:

                if row["AB"] > row["CV"] and row["AB"] > row["LMXRB"]:
                    df.loc[i, "AB"] += optical_score / 6

    return df