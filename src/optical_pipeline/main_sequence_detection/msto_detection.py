import numpy as np


def detect_msto(main_sequence_y: np.ndarray, 
                center_smooth: np.ndarray, 
                bins_division: int, 
                ymin: float, 
                MS_index_start: int) -> tuple[int, float]:
    """
    Detect the Main Sequence Turn-Off (MSTO) point from the smoothed
    main sequence ridge line.

    The MSTO is identified as the point where the main sequence stops
    decreasing in x-magnitude and begins to turn toward the red (i.e.,
    where the slope of the sequence changes sign).

    This is done by computing the discrete derivative of the smoothed
    central ridge (center_smooth) and locating the first transition
    from non-positive to positive slope.    
    """

    # --- Compute discrete derivative of the smoothed center line ---
    derivative = np.diff(center_smooth)

    negative = np.where(derivative <= 0)[0]

    if len(negative) == 0:
        
        return 0, main_sequence_y[0]
    

    # --- Get the first index where the derivative changes from negative to positive ---
    index_MSTO = next((i for i, x in enumerate(np.diff(np.append(negative, negative[-1]))) if x != 1), -1)
    
    offset = 1 / (2 * bins_division)

    msto_y = (index_MSTO + MS_index_start) / bins_division + ymin + offset

    return index_MSTO, msto_y