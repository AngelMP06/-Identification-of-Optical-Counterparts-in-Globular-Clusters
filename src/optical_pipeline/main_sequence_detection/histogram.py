import numpy as np
from ..utils.colors import get_color_columns


def compute_histogram(df, bins_division, graphic_comp):

    """
    Computing the 2D histogram for the given DataFrame and graphic components.
    """
    
    Color1, Color2 = get_color_columns(graphic_comp)

    x = df[f"{Color1} - {Color2}"]
    y = df[f"{Color2}_Mag"]

    xmax, xmin = round(x.max()+1), round(x.min()-1)
    ymax, ymin = round(y.max()+1), round(y.min()-1)

    # We define the bins and ranges for the histogram based on the data range and the desired division

    bins = [bins_division*(xmax-xmin), bins_division*(ymax-ymin)]
    ranges = [[xmin, xmax], [ymin, ymax]]

    H, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=ranges)

    return H, xmin, ymin