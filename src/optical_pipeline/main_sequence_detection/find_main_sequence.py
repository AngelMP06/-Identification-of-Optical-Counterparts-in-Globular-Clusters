import numpy as np
import pandas as pd

from .histogram import compute_histogram
from .ms_extraction import filter_histogram, get_max_width_bins
from .smoothing import smooth_sequence
from .msto_detection import detect_msto
from .ms_extraction import extract_main_sequence

from src.models import MainSequenceResults, Points, MSDetectionParams

def find_main_sequence(df: pd.DataFrame,
                       bins_division: int, 
                       CMD_config: int, 
                       ms_params: MSDetectionParams) -> tuple[MainSequenceResults, Points]:
    """
    Detect the main sequence (MS), its boundaries, MSTO and RGB.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with photometric data.
    bins_division : int
        Resolution of the CMD histogram.
    CMD_config : int
        Index defining the CMD configuration used.
    ms_detection : MSDetectionParams
        Parameters for main sequence detection.

    Returns
    -------
    main_sequence_results : MainSequenceResults
        Dataclass containing the main sequence detection results, including:
        (main_sequence_y : array-like
            Y-axis values of the MS.
        min_smooth : array-like
            Smoothed lower boundary of the MS.
        max_smooth : array-like
            Smoothed upper boundary of the MS.
        index_SGB : int
            Index marking the start of the Subgiant Branch.
        index_MSTO : int
            Index of the Main Sequence Turn-Off.
        msto_y : float
            Y-axis (magnitude) of the MSTO.)

    points : Points
        Dataclass containing key points used for classification, including:
        (point_G : list
            Point on the lower MS boundary at the faint end.
        point_H : list
            Point on the upper MS boundary at the faint end)
    """

    # --- Histogram ---
    H, xmin, ymin = compute_histogram(df, bins_division, CMD_config)


    # --- Clean histogram ---
    H = filter_histogram(H, ms_params.min_count)


    # --- Getting the max width for every horizontal line ---
    center_line, min_edge, max_edge = get_max_width_bins(H, ms_params.min_step)


    # --- Extract main sequence ---
    main_sequence_y, normalized_min, normalized_max, normalized_center, MS_index_start = extract_main_sequence(
        center_line, min_edge, max_edge, bins_division, xmin, ymin)


    # --- Smooth ---
    min_smooth, max_smooth, center_smooth = smooth_sequence(normalized_min, normalized_max, normalized_center)


    # --- MSTO detection ---
    index_MSTO, msto_y = detect_msto(main_sequence_y, center_smooth, bins_division, ymin, MS_index_start)


    # --- Limit the length of the MSTO and SGB to avoid misclassifications ---

    # Convert physical height → index space
    max_msto_idx = int(ms_params.maximum_lenght_MSTO * bins_division)


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

    main_sequence_y = np.asarray(main_sequence_y, dtype=np.float64)
    min_smooth = np.asarray(min_smooth, dtype=np.float64)
    max_smooth = np.asarray(max_smooth, dtype=np.float64)


    # --- Points G and H will be used later. ---
    point_G = np.array([max_smooth[index_MSTO], msto_y])
    point_H = np.array([min_smooth[index_MSTO], msto_y])

    main_sequence_results = MainSequenceResults(
        main_sequence_y = main_sequence_y,
        min_smooth = min_smooth,
        max_smooth = max_smooth,
        index_SGB = index_SGB,
        index_MSTO = index_MSTO,
        msto_y = msto_y
    )
    points = Points(
        point_G = point_G,
        point_H = point_H
    )

    return main_sequence_results, points
