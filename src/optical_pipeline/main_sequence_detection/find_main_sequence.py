import numpy as np

from .histogram import compute_histogram
from .ms_extraction import filter_histogram, get_max_width_bins
from .smoothing import smooth_sequence
from .msto_detection import detect_msto
from .ms_extraction import extract_main_sequence

def find_main_sequence(df, bins_division, graphic_comp, min_count, min_step, maximum_lenght_MSTO):
    """
    Detect the main sequence (MS), its boundaries, and MSTO.

    Returns:
        main_sequence_y
        min_smooth
        max_smooth
        index_SGB
        index_MSTO
        G, H points
    """

    # --- Histogram ---
    H, xmin, ymin = compute_histogram(df, bins_division, graphic_comp)

    # --- Clean histogram ---
    H = filter_histogram(H, min_count)



    # --- Getting the max width for every horizontal line ---
    center_line, min_edge, max_edge = get_max_width_bins(H, min_step)

    # --- Extract main sequence ---
    main_sequence_y, normalized_min, normalized_max, normalized_center, MS_index_start = extract_main_sequence(center_line, min_edge, max_edge, bins_division, xmin, ymin)

    # --- Smooth ---
    min_smooth, max_smooth, center_smooth = smooth_sequence(
        normalized_min, normalized_max, normalized_center, bins_division, xmin
    )

    # --- MSTO detection ---
    index_MSTO, msto_y = detect_msto(main_sequence_y, center_smooth, bins_division, ymin, MS_index_start)

    # --- Limit the length of the MSTO and SGB to avoid misclassifications ---

    # Convert physical height → index space
    max_msto_idx = int(maximum_lenght_MSTO * bins_division)

    # Define SGB start (half of MSTO window below MSTO)
    half_window = max_msto_idx // 2
    index_SGB = max(0, index_MSTO - half_window)

    # If MSTO is too deep in the sequence → trim lower part
    if index_MSTO > max_msto_idx:

        shift = index_MSTO - max_msto_idx

        # Trim arrays
        main_sequence_y = main_sequence_y[shift:]
        min_smooth = np.asarray(min_smooth)[shift:]
        max_smooth = np.asarray(max_smooth)[shift:]

        # Adjust indices
        index_MSTO -= shift
        index_SGB -= shift

    # --- Points G and H ---
    G = [max_smooth[index_MSTO], msto_y]
    H = [min_smooth[index_MSTO], msto_y]

    return main_sequence_y, min_smooth, max_smooth, index_SGB, index_MSTO, G, H
