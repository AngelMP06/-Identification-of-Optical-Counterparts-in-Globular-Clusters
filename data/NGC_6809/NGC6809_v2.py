import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import json



from scipy.signal import savgol_filter

def find_MS(optical_data_filtered, bins_division, gráfico_comp, min_count , min_step):

    if gráfico_comp == 0: #F275W-F336W
        Color1 = "275"
        Color2 = "336"
    elif gráfico_comp == 1: #F438W-F606W
        Color1 = "438"
        Color2 = "606"
    elif gráfico_comp == 2: #F606W-F814W
        Color1 = "606"
        Color2 = "814"
    
    S_Color2 = optical_data_filtered[Color2+"_Mag"]
    S_dif = optical_data_filtered[f"{Color1} - {Color2}"]

    xmax = round(S_dif.max()+1)
    xmin = round(S_dif.min()-1)
    ymax = round(S_Color2.max()+1)
    ymin = round(S_Color2.min()-1)

    bins_xy = [bins_division*(xmax-xmin), bins_division*(ymax-ymin)]
    range_xy = [[xmin, xmax], [ymin, ymax]]

    H, x_edges, y_edges = np.histogram2d(S_dif, S_Color2, bins = bins_xy, range = range_xy)
    
    for i in range(len(H)):
        for j in range(len(H[0])):
            if H[i][j] < min_count:
                H[i][j] = 0
    


    mean_MS = []       # mean
    min_MS = []     # min
    max_MS = []     # max
    for i in H.T:
        #First we go for every bin to find the position of the bins with stars, and we save the count of consecutive
        # bins with stars (ou) and the count of consecutive bins without stars (n).
        indexes_MS = []
        values_MS = []

        stored_indexes = []
        stored_values = []
        insured_indexes = []
        insured_values = []

        insured_width = 0
        width_MS = 0
        max_width_found = 0
        count_empty_bins = 0
        for j,z in enumerate(i):
            #We consider that a bin has stars if its count is greater than 0, and we save the position and count of the 
            # bins with stars in k and kk, and we save the position and count of the last consecutive bins with stars in 
            # ku and kkuu.
            if z > 0:
                count_empty_bins = 0
                width_MS += 1
                stored_indexes.append(j)     #position
                stored_values.append(z)    #value
                insured_indexes = []
                insured_values = []
                insured_width = 0
            else:
                count_empty_bins += 1
                if count_empty_bins < 2:
                    #The previus values are saved
                    insured_indexes = stored_indexes.copy()
                    insured_values = stored_values.copy()
                    insured_width = width_MS
                    #The new values are updated
                    width_MS += 1
                    stored_indexes.append(j)     #position
                    stored_values.append(z)    #value
                else:
                    if insured_width > max_width_found and insured_width >= min_step:
                        values_MS = insured_values
                        indexes_MS = insured_indexes
                        max_width_found = insured_width
                    stored_values = []
                    stored_indexes = []
                    width_MS = 0
                    insured_indexes = []
                    insured_values = []
                    insured_width = 0
        #We save the position of the left and right limits of the main sequence. If there are no stars in any bin, 
        # we save 0 as the limits.
        if len(indexes_MS)>0:
            min_MS.append(indexes_MS[0])
            max_MS.append(indexes_MS[-1])
        else:
            min_MS.append(0)
            max_MS.append(0)
        total = 0
        #We also save the mean position of the bins with stars to plot the mean line of the main sequence.
        if len(indexes_MS) > 0:
            for j in range(len(indexes_MS)):
                total += indexes_MS[j]*values_MS[j]
            mean_MS.append(total/sum(values_MS))
        else:
            mean_MS.append(0)

    

    non_zero_values_indexes = []
    for i,j in enumerate(mean_MS):
        if j == 0:
            non_zero_values_indexes.append(i)


    chain_width = 0  
    maximum_width = 0
    # Searching for the maximum separation between bins with value 0
    for i in range(len(non_zero_values_indexes)-1):
        index_difference = non_zero_values_indexes[i+1]-non_zero_values_indexes[i]-1
        if index_difference > maximum_width:
            maximum_width = index_difference
            chain_width = non_zero_values_indexes[i]
    main_sequence = range(chain_width+1,chain_width+maximum_width)  #Getting the maximum chain of indexes that have 0 as a value


    #Extract the positions (in cells) of the mean, max and min of the main sequence to plot them later. 
    filtered_x = [mean_MS[i] for i in main_sequence]
    filtered_max = [max_MS[i] for i in main_sequence]
    filtered_min = [min_MS[i] for i in main_sequence]

    #Change the positions from cells to magnitude values of the CMD
    normalized_min = (np.array(filtered_min)/bins_division)+xmin+1/(2*bins_division)
    normalized_max = (np.array(filtered_max)/bins_division)+xmin+1/(2*bins_division)
    normalized_x = (np.array(filtered_x)/bins_division)+xmin+1/(2*bins_division)

    main_sequence_y_magnitude = np.array(main_sequence)/bins_division+ymin+1/(2*bins_division)

    #Smooth the lines of the minimum and maximum of the main sequence using the Savitzky-Golay filter.
    window_size = int(len(normalized_min)/3)
    poly_order = 3
    min_smooth = savgol_filter(normalized_min, window_size, poly_order)
    max_smooth = savgol_filter(normalized_max, window_size, poly_order)
    x_smooth = savgol_filter(normalized_x, window_size, poly_order)

    #We obtain the position of the start of the main sequence turn off.
    derivative_x = []
    for i in range(len(x_smooth)-1):
        derivative_x.append(x_smooth[i]-x_smooth[i+1])

    container_d = []
    for i,j in enumerate(derivative_x):
        if j <= 0:
            container_d.append(i)
    if derivative_x[0] > 0 and 0 not in container_d:
        container_d.insert(0, -1)

    chain_width_d = 0
    maximum_d = 0
    for i in range(len(container_d)-1):
        m_d = container_d[i+1]-container_d[i]-1
        if m_d > maximum_d:
            maximum_d = m_d
            chain_width_d = container_d[i]
    msto_start = chain_width_d+maximum_d + chain_width + 1
    msto_start_y_magnitude = msto_start/bins_division+ymin+1/(2*bins_division)

    maximum_lenght_MSTO = 0.70 # The height of the MSTO and SGB can't be greater than 0.70.
    maximum_lenght_index_MSTO = int(maximum_lenght_MSTO * bins_division)
    half_lenght_index_MSTO = int(maximum_lenght_index_MSTO/2)

    index_MSTO_y_start = np.where(abs(main_sequence_y_magnitude - msto_start_y_magnitude) < 0.000001)[0][0]
    if index_MSTO_y_start < half_lenght_index_MSTO:
        index_SGB_y_start = 0
    else:
        index_SGB_y_start = index_MSTO_y_start-half_lenght_index_MSTO

    if index_MSTO_y_start > maximum_lenght_index_MSTO: 
        diference = index_MSTO_y_start - maximum_lenght_index_MSTO
        main_sequence_y_magnitude = np.delete(main_sequence_y_magnitude, range(diference))
        min_smooth = np.delete(min_smooth, range(diference))
        max_smooth = np.delete(max_smooth, range(diference))
        index_MSTO_y_start = index_MSTO_y_start - diference
        index_SGB_y_start = index_SGB_y_start - diference

    G = [max_smooth[np.where(msto_start_y_magnitude == main_sequence_y_magnitude)[0][0]], msto_start_y_magnitude]
    H = [min_smooth[np.where(msto_start_y_magnitude == main_sequence_y_magnitude)[0][0]], msto_start_y_magnitude]
    """
     # Plotting (optional)
    plt.scatter(S_dif, S_Color2, s = 5)
    plt.plot(min_smooth, main_sequence_y_magnitude, c = "black")
    plt.plot(max_smooth, main_sequence_y_magnitude, c = "black")
    plt.plot([S_dif.min(), S_dif.max()], [msto_start_y_magnitude, msto_start_y_magnitude], c = "red")
    plt.plot([S_dif.min(), S_dif.max()], [main_sequence_y_magnitude[index_SGB_y_start], main_sequence_y_magnitude[index_SGB_y_start]], c = "green")
    plt.gca().invert_yaxis()
    plt.show()
    """
    return main_sequence_y_magnitude, min_smooth, max_smooth, index_SGB_y_start, index_MSTO_y_start, G, H


def classify_position_MS(optical_data_filtered, gráfico_comp, main_sequence_y_magnitude, min_smooth, max_smooth, index_SGB_y_start, index_MSTO_y_start, significantly_bluer_limit, bluer_limit, redder_limit, significantly_redder_limit):
    if gráfico_comp == 0: #F275W-F336W
        Color1 = "275"
        Color2 = "336"
    elif gráfico_comp == 1: #F438W-F606W
        Color1 = "438"
        Color2 = "606"
    elif gráfico_comp == 2: #F606W-F814W
        Color1 = "606"
        Color2 = "814"
    df = optical_data_filtered
    df[f"position_{gráfico_comp}"] = "Unknown"

    y_col = Color2 + "_Mag"
    x_col = f"{Color1} - {Color2}"
    pos_col = f"position_{gráfico_comp}"

    y_vals = df[y_col].values
    x_vals = df[x_col].values

    # Initialize column once

    for i in range(len(main_sequence_y_magnitude) - 1):
        xmin_1, xmin_2 = min_smooth[i], min_smooth[i+1]
        xmax_1, xmax_2 = max_smooth[i], max_smooth[i+1]
        y_1, y_2 = main_sequence_y_magnitude[i], main_sequence_y_magnitude[i+1]

        slope = (xmin_2 - xmin_1) / (y_2 - y_1)

        # Mask for this segment (vectorized)
        mask = (y_vals >= y_1) & (y_vals < y_2)

        if not mask.any():
            continue

        y_seg = y_vals[mask]
        x_seg = x_vals[mask]

        x_min_real = xmin_1 + (y_seg - y_1) * slope
        x_max_real = xmax_1 + (y_seg - y_1) * slope

        # Classification (vectorized)
        if i <= index_SGB_y_start - 1:
            labels = np.select(
                [
                    x_seg < x_min_real - significantly_bluer_limit,
                    x_seg < x_min_real - bluer_limit,
                    x_seg < x_min_real,
                    x_seg < x_max_real,
                    x_seg < x_max_real + redder_limit,
                    x_seg < x_max_real + significantly_redder_limit,
                ],
                [
                    "bluer than MS L3",
                    "bluer than MS L2",
                    "bluer than SGB",
                    "Sub Giant Branch",
                    "redder than SGB",
                    "redder than MS L2",
                ],
                default="redder than MS L3",
            )

        elif i <= index_MSTO_y_start - 1:
            labels = np.select(
                [
                    x_seg < x_min_real - significantly_bluer_limit,
                    x_seg < x_min_real - bluer_limit,
                    x_seg < x_min_real,
                    x_seg < x_max_real,
                    x_seg < x_max_real + redder_limit,
                    x_seg < x_max_real + significantly_redder_limit,
                ],
                [
                    "bluer than MS L3",
                    "bluer than MS L2",
                    "bluer than MSTO",
                    "MSTO",
                    "redder than MSTO",
                    "redder than MS L2",
                ],
                default="redder than MS L3",
            )

        else:
            labels = np.select(
                [
                    x_seg < x_min_real - significantly_bluer_limit,
                    x_seg < x_min_real - bluer_limit,
                    x_seg < x_min_real,
                    x_seg < x_max_real,
                    x_seg < x_max_real + redder_limit,
                    x_seg < x_max_real + significantly_redder_limit,
                ],
                [
                    "bluer than MS L3",
                    "bluer than MS L2",
                    "bluer than MS L1",
                    "MS",
                    "redder than MS L1",
                    "redder than MS L2",
                ],
                default="redder than MS L3",
            )


        df.loc[mask, pos_col] = labels

    # Handle below MS (vectorized)
    df.loc[y_vals > main_sequence_y_magnitude[-1], pos_col] = "below the MS"

    """
    # Plotting (optional)
    sns.scatterplot(data = df, x = f"{Color1} - {Color2}", y = f"{Color2}_Mag", hue = f"position_{gráfico_comp}", s = 10)
    plt.gca().invert_yaxis()

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Region Classification")
    plt.legend()

    plt.show()
    """

    return df



def create_4_regions(main_sequence_y_magnitude, msto_start_y_magnitude, min_smooth, gráfico_comp, optical_data_filtered):
    if gráfico_comp == 0: #F275W-F336W
        Color1 = "275"
        Color2 = "336"
    elif gráfico_comp == 1: #F438W-F606W
        Color1 = "438"
        Color2 = "606"
    elif gráfico_comp == 2: #F606W-F814W
        Color1 = "606"
        Color2 = "814"    
    # --- Precompute once ---
    y_vals = main_sequence_y_magnitude
    idx_A = np.argmax(y_vals == msto_start_y_magnitude)  # faster than np.where()[0][0]

    x_A = min_smooth[idx_A]
    y_A = msto_start_y_magnitude

    x_B = min_smooth[0]
    y_B = y_vals[0]

    Min_MS = y_vals[-1]

    slope_m = (x_B - x_A) / (y_B - y_A)

    # Column names
    x_col = f"{Color1} - {Color2}"
    y_col = f"{Color2}_Mag"
    pos_col = f"position_{gráfico_comp}"

    # --- Work directly on arrays (FASTER) ---
    df = optical_data_filtered
    x = df[x_col].values
    y = df[y_col].values
    pos = df[pos_col].values

    # Precompute line (vectorized)
    line = x_A + slope_m * (y - y_A)

    # --- Masks (no copies yet) ---
    mask_r1 = y >= Min_MS

    mask_r2 = (y < Min_MS) & (y >= y_A)

    mask_common_upper = (y < y_A) & (y >= y_B)
    mask_lower = y < y_B

    mask_sgb_msto = (pos == "Sub Giant Branch") | (pos == "MSTO")

    mask_r3 = (
        (mask_common_upper & (x > line)) |
        (mask_lower & (x > x_B)) |
        mask_sgb_msto
    )

    mask_r4 = (
        ((mask_common_upper & (x <= line)) |
        (mask_lower & (x <= x_B))) &
        (~mask_sgb_msto)
    )



    # --- Only now create DataFrames (if needed) ---
    region1 = df[mask_r1]
    region2 = df[mask_r2]
    region3 = df[mask_r3]
    region4 = df[mask_r4]
    """
    # Plotting (optional)
    plt.scatter(region1[x_col], region1[y_col], s=0.5, label='R1')
    plt.scatter(region2[x_col], region2[y_col], s=0.5, label='R2')
    plt.scatter(region3[x_col], region3[y_col], s=0.5, label='R3')
    plt.scatter(region4[x_col], region4[y_col], s=0.5, label='R4')
    plt.plot([x_A, x_B], [y_A, y_B], c = "black")

    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()
    """

    return region1, region2, region3, region4



from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

def HB_classification(region4, gráfico_comp):
    if gráfico_comp == 0: #F275W-F336W
        Color1 = "275"
        Color2 = "336"
    elif gráfico_comp == 1: #F438W-F606W
        Color1 = "438"
        Color2 = "606"
    elif gráfico_comp == 2: #F606W-F814W
        Color1 = "606"
        Color2 = "814"  

    df = region4.copy()

    x = df[f"{Color1} - {Color2}"]
    y = df[f"{Color2}_Mag"]

    X = np.column_stack((x, y))

    # Normalize
    X_scaled = StandardScaler().fit_transform(X)

    db = DBSCAN(eps=0.4, min_samples=15).fit(X_scaled)
    labels = db.labels_

    unique_labels = set(labels)

    # --- Extract clusters (ignore noise) ---
    clusters = [label for label in unique_labels if label != -1]

    # --- Case 1: Only one cluster ---
    if len(clusters) == 1:
        selected_label = clusters[0]

    # --- Case 2: More than one cluster ---
    else:
        best_label = None
        best_score = None

        for label in clusters:
            mask = labels == label
            
            x_mean = np.mean(x[mask])
            y_mean = np.mean(y[mask])
            
            # We want TOP-LEFT:
            # smaller x → more left
            # smaller y → brighter (top in CMD)
            score = 2*x_mean + y_mean  # minimize this
            
            if best_score is None or score < best_score:
                best_score = score
                best_label = label

        selected_label = best_label
    mask = labels == selected_label

    x_selected = x[mask].values
    y_selected = y[mask].values

    C = [x_selected.min(), y_selected.max()]
    D = [x_selected.max(), y_selected.min()]
    E = [x_selected.max(), y_selected.max()]

    df.loc[labels == selected_label, f"position_{gráfico_comp}"] = "HB"
    """
    # Plotting (optional)
    sns.scatterplot(data = df, x = f"{Color1} - {Color2}", y = f"{Color2}_Mag", hue = f"position_{gráfico_comp}", s = 10)
    plt.gca().invert_yaxis()

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Region Classification")
    plt.legend()

    plt.show()
    """
    return df, C, D, E

from scipy.interpolate import interp1d

def GB_classification(region3, gráfico_comp, bins_division, msto_start_y_magnitude):
    if gráfico_comp == 0: #F275W-F336W
        Color1 = "275"
        Color2 = "336"
    elif gráfico_comp == 1: #F438W-F606W
        Color1 = "438"
        Color2 = "606"
    elif gráfico_comp == 2: #F606W-F814W
        Color1 = "606"
        Color2 = "814" 

    # We want the most bottom-right cluster, it is a secure part of the RGB region
    x = region3[f"{Color1} - {Color2}"]
    y = region3[f"{Color2}_Mag"]

    X = np.column_stack((x, y))

    # Normalize
    X_scaled = StandardScaler().fit_transform(X)

    db = DBSCAN(eps=0.4, min_samples=15).fit(X_scaled)
    labels = db.labels_

    unique_labels = set(labels)

    # --- Extract clusters (ignore noise) ---
    clusters = [label for label in unique_labels if label != -1]

    # --- Case 1: Only one cluster ---
    if len(clusters) == 1:
        selected_label = clusters[0]

    # --- Case 2: More than one cluster ---
    else:
        best_label = None
        best_score = None

        for label in clusters:
            mask = labels == label
            
            x_mean = np.mean(x[mask])
            y_mean = np.mean(y[mask])
            
            # We want the BOTTOM-LEFT cluster so we minimize x-y
            score = x_mean - y_mean  # minimize this
            
            if best_score is None or score < best_score:
                best_score = score
                best_label = label

        selected_label = best_label



    # --- Extract coordinates of selected cluster ---
    mask = labels == selected_label

    x_selected = x[mask].values
    y_selected = y[mask].values

    #We need the most top-right point of the cluster to define the AGB and Red Stragglers.
    F = [x_selected.max(), y_selected.min()]

    region3.loc[(labels == selected_label) & (region3[f"position_{gráfico_comp}"] == "Unknown"), f"position_{gráfico_comp}"] = "RGB"

    xmax = round(x_selected.max()+1)
    xmin = round(x_selected.min()-1)
    ymax = round(y_selected.max()+1)
    ymin = round(y_selected.min()-1)

    bins_xy = [int(bins_division*(xmax-xmin)/4), int(bins_division*(ymax-ymin)/4)]

    H, xedges, yedges = np.histogram2d(x_selected, y_selected, bins = bins_xy,)

    # Bin centers
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])

    # Ridge extraction (same as before)
    ridge_x, ridge_y = [], []

    for i in range(len(x_centers)):
        column = H[:, i]
        
        if np.sum(column) == 0:
            continue
        
        max_idx = np.argmax(column)
        ridge_x.append(x_centers[i])
        ridge_y.append(y_centers[max_idx])

    ridge_x = np.array(ridge_x)
    ridge_y = np.array(ridge_y)
    ridge_complete_x = np.concatenate(([x_selected.min()], ridge_x, [F[0]]))
    ridge_complete_y = np.concatenate(([msto_start_y_magnitude], ridge_y, [F[1]]))


    # Interpolate ridge
    # ---------------------------
    ridge_func = interp1d(
        ridge_complete_x, ridge_complete_y,
        kind='linear',
        bounds_error=False,
        fill_value="extrapolate"
    )

    #  Define vertical boundary
    x_cut = F[0]

    # Classify points
    pos = []

    for xi, yi in zip(x, y):
        
        if xi > x_cut:
            pos.append("Red Super Giant Branch")
        else:
            y_ridge = ridge_func(xi)
            
            if yi > y_ridge:
                pos.append("Red Straggler")
            else:
                pos.append("Unknown")
    for i,j in enumerate(region3.index):
        if region3.loc[j,f"position_{gráfico_comp}"] == "Unknown":
            region3.loc[j,f"position_{gráfico_comp}"] = pos[i]
        if region3.loc[j,f"position_{gráfico_comp}"] == "redder than MSTO" or region3.loc[j,f"position_{gráfico_comp}"] == "bluer than MSTO":
            region3.loc[j,f"position_{gráfico_comp}"] = "MSTO"
        if region3.loc[j,f"position_{gráfico_comp}"] == "redder than SGB" or region3.loc[j,f"position_{gráfico_comp}"] == "bluer than SGB":
            region3.loc[j,f"position_{gráfico_comp}"] = "Sub Giant Branch"
    """
    # Plotting (optional)
    sns.scatterplot(data = region3, x = f"{Color1} - {Color2}", y = f"{Color2}_Mag", hue = f"position_{gráfico_comp}")
    plt.gca().invert_yaxis()

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Region Classification")
    plt.legend()

    plt.show()
    """
    return region3, F
    

from matplotlib.path import Path

def AGB_BS_classification(df, gráfico_comp, C, D, E, F, G, H):
    if gráfico_comp == 0: #F275W-F336W
        Color1 = "275"
        Color2 = "336"
    elif gráfico_comp == 1: #F438W-F606W
        Color1 = "438"
        Color2 = "606"
    elif gráfico_comp == 2: #F606W-F814W
        Color1 = "606"
        Color2 = "814" 
    #Defining Blue Straglers inside the polygon C, D, G, H.
    polygon = np.array([
    [C[0], C[1]],
    [D[0], D[1]],
    [G[0], G[1]],
    [H[0], H[1]]
    ])

    poly_path = Path(polygon)
    points = np.column_stack((
    df[f"{Color1} - {Color2}"],
    df[f"{Color2}_Mag"]
    ))

    inside = poly_path.contains_points(points)

    for i in df.index:
        if df.loc[i, f"position_{gráfico_comp}"] == "Unknown" and inside[i] == True: 
            df.loc[i, f"position_{gráfico_comp}"] = "BS"

    #Defining AGB stars as those above the line defined by points E and F.
    m = (F[1] - E[1]) / (F[0] - E[0])
    b = E[1] - m * E[0]

    x_vals = df[f"{Color1} - {Color2}"]
    y_vals = df[f"{Color2}_Mag"]
    pos_vals = df[f"position_{gráfico_comp}"]

    mask_agb = (x_vals > E[0]) & (y_vals < (m * x_vals + b)) & (pos_vals == "Unknown")

    df.loc[mask_agb, f"position_{gráfico_comp}"] = "AGB"

    """

    sns.scatterplot(data = df, x = f"{Color1} - {Color2}", y = f"{Color2}_Mag", hue = f"position_{gráfico_comp}", s = 10)
    plt.gca().invert_yaxis()
    plt.plot(np.append(polygon, [[C[0], C[1]]], axis = 0)[:,0], np.append(polygon, [[C[0], C[1]]], axis = 0)[:,1], c = "blue")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Region Classification")
    plt.legend()
    x_plot = np.linspace(x_vals.min(), x_vals.max(), 200)
    y_plot = m * x_plot + b

    plt.plot(x_plot, y_plot, color='red', linewidth=2)
    plt.axvline(E[0], color='red', linestyle='--')

    plt.scatter([E[0], F[0], C[0], D[0], G[0], H[0]], [E[1], F[1], C[1], D[1], G[1], H[1]], color='black')
    plt.show()

    """

    return df


def generate_optical_file(path_optical, path_optical_arrival, distance_parsecs, Mag_min = -50, Mag_max = 50, RMS_max = 1, Fit_max = 1.5, Sharp_min = -2, Sharp_max = 2, CM_min = 80, bins_division = 20, min_count = 2, min_step = 2, bluer_limit = 0.5, significantly_bluer_limit = 1.0, redder_limit = 0.5, significantly_redder_limit = 1.0):
    
    optical_data = pd.read_csv(
        path_optical,
        sep=r"\s+",      # split by any amount of whitespace
        header=None      # because there are no column names
    )
    optical_data.columns = [
            "X", "Y",
            "275_Mag", "275_RMS", "275_Fit", "275_Sharp", "275_exp_found", "275_exp_well",
            "336_Mag", "336_RMS", "336_Fit", "336_Sharp", "336_exp_found", "336_exp_well",
            "438_Mag", "438_RMS", "438_Fit", "438_Sharp", "438_exp_found", "438_exp_well",
            "606_Mag", "606_RMS", "606_Fit", "606_Sharp", "606_exp_found", "606_exp_well",
            "814_Mag", "814_RMS", "814_Fit", "814_Sharp", "814_exp_found", "814_exp_well",
            "Cluster_Membership",
            "RA", "Decl",
            "Id",
            "Iteration_found"
    ]

    def filter_data(optical_data,distance_parsecs, Mag_min, Mag_max, RMS_max, Fit_max, Sharp_min, Sharp_max, CM_min):
        optical_data["Visible"] = (optical_data["438_Mag"]+optical_data["606_Mag"])/2

        optical_data["Mv"] = optical_data["Visible"]-5*(math.log(distance_parsecs/10,10))

        bands = ["275", "336", "438", "606", "814"]
        mask = np.ones(len(optical_data), dtype=bool)
        for b in bands:
            mask &= optical_data[f"{b}_Mag"].between(Mag_min, Mag_max)
            mask &= optical_data[f"{b}_RMS"] < RMS_max
            mask &= optical_data[f"{b}_Fit"] < Fit_max
            mask &= optical_data[f"{b}_Sharp"].between(Sharp_min, Sharp_max)
            mask &= optical_data["Cluster_Membership"] > CM_min


        optical_data_filtered = optical_data[mask].copy()
        optical_data_filtered.reset_index(inplace = True)
        column_drop = ["X", "Y", "Iteration_found", "index"]
        for i in bands:
            column_drop.append(i+"_RMS")
            column_drop.append(i+"_Fit")
            column_drop.append(i+"_Sharp")
            column_drop.append(i+"_exp_found")
            column_drop.append(i+"_exp_well")
        optical_data_filtered.drop(columns = column_drop, inplace = True)

        optical_data_filtered.sort_index(inplace = True)
        optical_data_filtered['275 - 336'] = optical_data_filtered["275_Mag"] - optical_data_filtered["336_Mag"]
        optical_data_filtered['438 - 606'] = optical_data_filtered["438_Mag"] - optical_data_filtered["606_Mag"]
        optical_data_filtered['606 - 814'] = optical_data_filtered["606_Mag"] - optical_data_filtered["814_Mag"]

        #print(optical_data_filtered)

        return optical_data_filtered

    gráfico_comp = 2
    #for gráfico_comp in range(3):
    if gráfico_comp == 0: #F275W-F336W
        Color1 = "275"
        Color2 = "336"
    elif gráfico_comp == 1: #F438W-F606W
        Color1 = "438"
        Color2 = "606"
    elif gráfico_comp == 2: #F606W-F814W
        Color1 = "606"
        Color2 = "814"

    optical_data_filtered = filter_data(optical_data, distance_parsecs, Mag_min, Mag_max, RMS_max, Fit_max, Sharp_min, Sharp_max, CM_min)
    
    main_sequence_y_magnitude, min_smooth, max_smooth, index_SGB_y_start, index_MSTO_y_start, point_G, point_H = find_MS(optical_data_filtered, bins_division, gráfico_comp, min_count , min_step)
 
    optical_data_classified_MS = classify_position_MS(optical_data_filtered, gráfico_comp, main_sequence_y_magnitude, min_smooth, max_smooth, index_SGB_y_start, index_MSTO_y_start, significantly_bluer_limit, bluer_limit, redder_limit, significantly_redder_limit)
    
    region1, region2, region3, region4 = create_4_regions(main_sequence_y_magnitude, main_sequence_y_magnitude[index_MSTO_y_start], min_smooth, gráfico_comp, optical_data_classified_MS)
    
    region4, point_C, point_D, point_E = HB_classification(region4, gráfico_comp)
    
    region3, point_F = GB_classification(region3, gráfico_comp, bins_division, main_sequence_y_magnitude[index_MSTO_y_start])
    
    df_merged = pd.concat([region1, region2, region3, region4]).sort_index()
    
    df_final = AGB_BS_classification(df_merged, gráfico_comp, point_C, point_D, point_E, point_F, point_G, point_H)
    """
    #df_final.to_csv(path_optical_arrival, index=False)
    """
generate_optical_file(distance_parsecs = 5400, path_optical = 'C:/Users/HP/Desktop/Programación/Workspace/Python/Astronomy/NGC 6809/NGC6809.txt', path_optical_arrival = 'C:/Users/HP/Desktop/Programación/Workspace/Python/Astronomy/NGC 6809/NGC6809_optical.csv')


"""
Mag_min = -50,  # 0
Mag_max = 50   # 35,
RMS_max = 1,     # 0.1 #No quiero que aparezcan varias estrellas dentro del circulo, pues si eso sucede de todas formas no será posible dar un valor correcto
Fit_max = 1.5,   # También puede ser 0.6, pero en en los filtros 606 y 814 no se pueden usar valores tan estrictos, pues se eliminarían demasiados puntos
Sharp_min = -2,  # -0.25
Sharp_max = 2,   # 0.25
CM_min = 80,
bins_division = 20,
min_count = 2,
min_step = 2,
bluer_limit = 0.5,
significantly_bluer_limit = 1.0
redder_limit = 0.5,
significantly_redder_limit = 1.0)
"""