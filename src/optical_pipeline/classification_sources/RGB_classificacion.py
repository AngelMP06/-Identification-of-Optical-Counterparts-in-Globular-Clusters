import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from ..utils.colors import get_color_columns


def DBSCAN_classification(region3, graphic_comp, eps=0.4, min_samples=15):

    """
    Apply DBSCAN clustering in the CMD and label the bottom-left cluster as the RGB.
    """

    region3 = region3.copy()

    # --- Colors ---
    Color1, Color2 = get_color_columns(graphic_comp)

    x_col = f"{Color1} - {Color2}"
    y_col = f"{Color2}_Mag"
    pos_col = f"position_{graphic_comp}"

    x = region3[x_col].values
    y = region3[y_col].values

    # --- DBSCAN clustering ---
    X = np.column_stack((x, y))
    X_scaled = StandardScaler().fit_transform(X)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled).labels_
    clusters = [l for l in set(labels) if l != -1]

    if len(clusters) == 0:
        return region3, None, None, None

    # --- Select RGB cluster (bottom-right) ---
    if len(clusters) == 1:
        selected = clusters[0]
    else:
        best_label = None
        best_score = None

        for label in clusters:
            mask = labels == label
            x_mean = np.mean(x[mask])
            y_mean = np.mean(y[mask])

            # bottom-right → large x, large y
            score = x_mean - y_mean

            if best_score is None or score < best_score:
                best_score = score
                best_label = label

        selected = best_label

    mask_rgb = labels == selected

    x_sel = x[mask_rgb]
    y_sel = y[mask_rgb]

    # --- Key point F ---
    point_F = [x_sel.max(), y_sel.min()]

    # --- Assign RGB ---
    region3.loc[mask_rgb & (region3[pos_col] == "Unknown"), pos_col] = "RGB"
    return region3, point_F, x_sel, y_sel

def ridge_line_extraction(x_sel, y_sel, bins_division, msto_y, point_F):


    """
    Extract the central ridge line of the Red Giant Branch (RGB) using a 2D histogram.
    This ridge line is extender to cover the bottom left point of the histogram and the point F.
    """

# --- Histogram for ridge ---
    xmin, xmax = x_sel.min(), x_sel.max()
    ymin, ymax = y_sel.min(), y_sel.max()

    bins = [
        int(bins_division * (xmax - xmin) / 4),
        int(bins_division * (ymax - ymin) / 4)
    ]

    H, xedges, yedges = np.histogram2d(x_sel, y_sel, bins=bins)

    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])

    # --- Ridge extraction ---
    ridge_x = []
    ridge_y = []

    for i in range(len(x_centers)):
        column = H[:, i]
        if column.sum() == 0:
            continue

        idx = np.argmax(column)
        ridge_x.append(x_centers[i])
        ridge_y.append(y_centers[idx])

    ridge_x = np.array(ridge_x)
    ridge_y = np.array(ridge_y)

    # --- Extend ridge ---
    ridge_x_full = np.concatenate(([xmin], ridge_x, [point_F[0]]))
    ridge_y_full = np.concatenate(([msto_y], ridge_y, [point_F[1]]))

    return ridge_x_full, ridge_y_full
    
def SGB_RS_classification(region3, graphic_comp, ridge_x_full, ridge_y_full, point_F):
    
    """
    Classify the sources as Red Supergiant Branch (RSGB) if they are to the right of point F, 
    and as Red Lagging if they are to the left of point F and below the ridge line.
    """

    region3 = region3.copy()

    # --- Colors ---
    Color1, Color2 = get_color_columns(graphic_comp)
    y_col= f"{Color2}_Mag"
    x_col = f"{Color1} - {Color2}"

    x = region3[x_col].values
    y = region3[y_col].values
    pos_col = f"position_{graphic_comp}"
    
    
    # --- Interpolation ---
    ridge_func = interp1d(
        ridge_x_full,
        ridge_y_full,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate" # type: ignore
    )

    # --- Vectorized classification ---
    y_ridge = ridge_func(x)

    # --- Classify Red Super Giant Branch (RSGB) sourecs as the ones at the right of the point F ---
    mask_super = x > point_F[0] 

    # --- Classify the sources below the ridge line as Red Stragglers (RS) (Remember that the y axis is inverted)---
    mask_straggler = (x <= point_F[0]) & (y > y_ridge)

    # Apply only where still Unknown
    unknown_mask = region3[pos_col] == "Unknown"

    region3.loc[unknown_mask & mask_super, pos_col] = "Red Super Giant Branch"
    region3.loc[unknown_mask & mask_straggler, pos_col] = "Red Straggler"


    return region3

def RGB_classification(region3, graphic_comp, bins_division, msto_y, eps=0.4, min_samples=15):
    """
    Analize sources of region 3 of the CMD to identify the Red Giant Branch (RGB), Red Stragglers (RS), and Red Super Giant Branch (RSGB).

    Steps:
    - Apply DBSCAN clustering in CMD and label the bottom-left cluster as the RGB.
    - Use a 2d histogram to identify all the RGB sources.
    - Define the point F as the top-right boundary of the RGB.
    - Compute the central ridge line of the RGB and classify the sources below this line as Red Stragglers (RS).
    - Classify Red Super Giant Branch (RSGB) sourecs as the ones at the right of the point F.
    """
    region3 = region3.copy()

    # --- Step 1: DBSCAN classification ---
    region3, point_F, x_sel, y_sel = DBSCAN_classification(region3, graphic_comp, eps, min_samples)

    if point_F is None:
        return region3, None
    
    # --- Step 2: Ridge line extraction ---
    ridge_x_full, ridge_y_full = ridge_line_extraction(x_sel, y_sel, bins_division, msto_y, point_F)

    # --- Step 3: SGB and RS classification ---
    region3 = SGB_RS_classification(region3, graphic_comp, ridge_x_full, ridge_y_full, point_F)

    return region3, point_F

