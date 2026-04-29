import numpy as np

def compute_base_probability(df):

    """
    Compute the initial probability of being a true counterpart.

    This function combines the class scores (AB, CV, LMXRB) into a total
    counterpart probability for each candidate.

    It computes the total probability for each x-ray source by substracting to 1
    the probability that none of its candidates are the true counterpart.

    """

    df = df.copy()
    # Calculating the probability for not being a counterpart
    df["p_counterpart"] = df["AB"] + df["CV"] + df["LMXRB"]
    df["p_not"] = 1 - df["p_counterpart"]

    # The total probability for each x-ray source is 1 minus the probability that none of its candidates are the true counterpart
    total_prob = 1 - df.groupby("Xray source")["p_not"].prod()

    return df, total_prob

def apply_weighting(df):

    """
    Enhance probabilities based on classification confidence.

    This step increases the weight of candidates with a clear dominant class.
    The weighting is based on the ratio between the highest and second-highest
    class scores, favoring sources with less ambiguous classifications.
    """

    df = df.copy()
    
    weights = []
    
    for _, row in df.iterrows():
        
        # For each row we create an array with the scores of being AB, CV, LMXRB 
        # and we sort it in descending order.
        arr = np.array([row["AB"], row["CV"], row["LMXRB"]])
        arr_sorted = np.sort(arr)[::-1]

        # We compute the ratio between the highest and second-highest score. 
        if arr_sorted[1] != 0:
            ratio = arr_sorted[0] / arr_sorted[1]
        else:
            ratio = 1e9

        # We weight the probability of being a counterpart by this ratio, so that 
        # sources with a clear dominant class receive a higher probability.
        weights.append(row["p_counterpart"] * ratio)

    df["p_weighted"] = weights

    return df

def normalize_probabilities(df, total_prob):

    """
    Normalize probabilities within each X-ray source.

    The weighted probabilities are rescaled so that the total probability
    distribution per X-ray source is consistent with the total_prob value
    found in compute_base_probability.
    """

    df = df.copy()

    # Calculating the total weighted probability for each x-ray source
    sum_after = df.groupby("Xray source")["p_weighted"].sum()

    new_probs = []

    # For each candidate, we normalize the probability so the sum of the 
    # probabilities of all candidates for a given x-ray source equals the 
    # total probability of that x-ray source.
    for i, row in df.iterrows():

        x_id = row["Xray source"]

        p = row["p_weighted"] * total_prob.loc[x_id] / sum_after.loc[x_id]
        new_probs.append(p)

    df["p_final"] = new_probs

    return df

def apply_radius_penalty(df):

    """
    Penalize candidates with less reliable positional matches.

    Candidates that are not identified in the 95% confidence radius and
    are identified using the fallback search radius (2 arcsec)
    receive a reduction in their final probability.
    """

    df = df.copy()

    df.loc[df["radius"] == "2", "p_final"] *= 0.8

    return df

def rescale_classes(df):

    """
    Rescale class probabilities to match final counterpart likelihood.

    After computing the final probability for each candidate, the class
    scores (AB, CV, LMXRB) are proportionally adjusted so that their sum
    remains consistent with the updated counterpart probability.
    """
    
    df = df.copy()

    scale = df["p_final"] / df["p_counterpart"]

    df["AB"] *= scale
    df["CV"] *= scale
    df["LMXRB"] *= scale

    return df