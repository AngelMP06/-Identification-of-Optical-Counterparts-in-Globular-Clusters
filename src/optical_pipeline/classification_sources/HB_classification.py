import numpy as np
import pandas as pd
    
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from src.models import Points

from ..utils.colors import get_color_columns

def HB_classification(
    region4: pd.DataFrame,
    points: Points,
    CMD_config: int,
    eps=0.5,
    min_samples=15
    )-> tuple[pd.DataFrame, Points]:
    """
    Take the region 4 and identify Horizontal Branch (HB) stars using DBSCAN clustering.

    Steps:
    - Apply DBSCAN clustering in CMD space
    - Select the cluster located at the top-left (bright + blue)
    - Label that cluster as HB
    - Return bounding points C, D, E
    """

    region4 = region4.copy()

    # --- Colors ---
    Color1, Color2 = get_color_columns(CMD_config)

    x_col = f"{Color1} - {Color2}"
    y_col = f"{Color2}_Mag"
    pos_col = f"position_{CMD_config}"

    x = np.array(region4[x_col].values)
    y = np.array(region4[y_col].values)


    # --- Prepare data ---
    X = np.column_stack((x, y))
    X_scaled = StandardScaler().fit_transform(X)


    # --- Clustering ---
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    labels = db.labels_


    # --- Get valid clusters (exclude noise) ---
    clusters = [l for l in set(labels) if l != -1]


    # --- Edge case: no clusters ---
    if len(clusters) == 0:
        return region4, points


    # --- Select cluster ---
    if len(clusters) == 1:
        selected_label = clusters[0]
    else:
        best_label = None
        best_score = None

        for label in clusters:
            mask = labels == label

            x_mean = np.mean(x[mask])
            y_mean = np.mean(y[mask])

            # TOP-LEFT → minimize x and y
            score = 2 * x_mean + y_mean

            if best_score is None or score < best_score:
                best_score = score
                best_label = label

        selected_label = best_label


    # --- Extract selected cluster ---
    mask = labels == selected_label

    x_sel = x[mask]
    y_sel = y[mask]


    # --- Define bounding points, they will be used later ---
    point_C = np.array([x_sel.min(), y_sel.max()])
    point_D = np.array([x_sel.max(), y_sel.min()])
    point_E = np.array([x_sel.max(), y_sel.max()])


    # --- Assign labels ---
    region4.loc[mask, pos_col] = "HB"
    
    points.point_C = point_C
    points.point_D = point_D
    points.point_E = point_E

    return region4, points