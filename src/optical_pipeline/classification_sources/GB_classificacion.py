import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from ..utils.colors import get_color_columns


def GB_classification(
    region3,
    graphic_comp,
    bins_division,
    msto_y,
    eps=0.4,
    min_samples=15
):
    """
    Classify the Red Giant Branch (RGB), Red Stragglers (RS), and Super Giants Branch (SGB) in region3.
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
        return region3, None

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

    mask_super = x > point_F[0] 
    mask_straggler = (x <= point_F[0]) & (y > y_ridge)
    mask_unknown = ~(mask_super | mask_straggler)

    # Apply only where still Unknown
    unknown_mask = region3[pos_col] == "Unknown"

    region3.loc[unknown_mask & mask_super, pos_col] = "Red Super Giant Branch"
    region3.loc[unknown_mask & mask_straggler, pos_col] = "Red Straggler"

    # --- Normalize MSTO / SGB labels ---
    region3.loc[region3[pos_col].isin(["redder than MSTO", "bluer than MSTO"]), pos_col] = "MSTO"
    region3.loc[region3[pos_col].isin(["redder than SGB", "bluer than SGB"]), pos_col] = "Sub Giant Branch"

    return region3, point_F