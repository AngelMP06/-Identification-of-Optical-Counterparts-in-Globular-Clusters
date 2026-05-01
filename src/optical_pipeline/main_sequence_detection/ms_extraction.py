import numpy as np


def filter_histogram(H: np.ndarray, min_count: int) -> np.ndarray:
    """
    Remove low-density bins from histogram.
    """
    H = H.copy()
    H[H < min_count] = 0
    return H


def get_max_width_bins(H: np.ndarray, min_step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the max width where there are counts for each horizontal line in the histogram.

    - Allows small gaps (1 empty bin tolerance)
    - Tracks consecutive regions
    - Uses 'insured' segments to recover sequences
    - Selects the widest valid MS segment

    And returns center line, left edge and right edge of the MAin Sequence in bins of the 2d histogram.
    """

    center_line = []
    min_edge = []
    max_edge = []

    for column in H.T:

        indexes = []
        values = []

        stored_idx = []
        stored_val = []

        insured_idx = []
        insured_val = []

        width = 0
        insured_width = 0
        max_width = 0
        empty_count = 0

        for i, val in enumerate(column):

            if val > 0:
                empty_count = 0
                width += 1

                stored_idx.append(i)
                stored_val.append(val)

                insured_idx = []
                insured_val = []
                insured_width = 0

            else:
                empty_count += 1

                if empty_count < 2:
                    # Save previous valid segment
                    insured_idx = stored_idx.copy()
                    insured_val = stored_val.copy()
                    insured_width = width

                    # Extend segment including gap
                    width += 1
                    stored_idx.append(i)
                    stored_val.append(val)

                else:
                    # Evaluate insured segment
                    if insured_width > max_width and insured_width >= min_step:
                        indexes = insured_idx
                        values = insured_val
                        max_width = insured_width

                    # Reset everything
                    stored_idx = []
                    stored_val = []
                    insured_idx = []
                    insured_val = []
                    width = 0
                    insured_width = 0

        # Final assignment
        if len(indexes) > 0:
            min_edge.append(indexes[0])
            max_edge.append(indexes[-1])

            weighted_mean = np.average(indexes, weights=values)
            center_line.append(weighted_mean)
        else:
            min_edge.append(0)
            max_edge.append(0)
            center_line.append(0)

    center_line = np.array(center_line)
    min_edge = np.array(min_edge)
    max_edge = np.array(max_edge)

    return center_line, min_edge, max_edge

def extract_main_sequence(center_line: np.ndarray, 
                          min_edge: np.ndarray, 
                          max_edge: np.ndarray, 
                          bins_division: int, 
                          xmin: float, 
                          ymin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Filters the Main Sequence bins using the center line and returns the values in magnitude.
    The main sequence is defined as the longest contiguous segment of non-zero values in the center line.
    """

    # --- Find zero positions ---
    zero_idx = np.where(center_line == 0)[0]

    # Edge case: no zeros → entire range is MS
    if len(zero_idx) < 2:
        start = 0
        main_sequence = np.arange(len(center_line))
    else:


        # --- Find largest gap between zeros ---
        gaps = zero_idx[1:] - zero_idx[:-1] - 1

        max_gap_idx = np.argmax(gaps)
        max_width = gaps[max_gap_idx]
        start = zero_idx[max_gap_idx]

        main_sequence = np.arange(start + 1, start + max_width)


    # --- Extract MS in bin space ---
    filtered_center = center_line[main_sequence]
    filtered_min = min_edge[main_sequence]
    filtered_max = max_edge[main_sequence]


    # --- Convert to magnitude space ---
    offset = 1 / (2 * bins_division)

    normalized_min = filtered_min / bins_division + xmin + offset
    normalized_max = filtered_max / bins_division + xmin + offset
    normalized_center = filtered_center / bins_division + xmin + offset

    main_sequence_y = main_sequence / bins_division  + ymin + offset

    MS_index_start = start + 1

    return main_sequence_y, normalized_min, normalized_max, normalized_center, MS_index_start