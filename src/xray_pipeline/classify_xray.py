import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from .module_xray import (
    load_xray_data,
    parse_coordinates,
    compute_counts_and_errors,
    compute_luminosity,
    classify_sources,
    finalize_xray
)

def x_ray_classification(path_xray, BS_RA, BS_Decl, distance_parsecs):
    df = load_xray_data(path_xray)
    df = parse_coordinates(df, BS_RA, BS_Decl)
    df = compute_counts_and_errors(df)
    df = compute_luminosity(df, distance_parsecs)
    df = classify_sources(df)
    df = finalize_xray(df)  

    print("X-ray classification completed")

    return df






