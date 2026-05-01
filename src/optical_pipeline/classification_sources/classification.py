import pandas as pd

from .RGB_classificacion import RGB_classification
from .HB_classification import HB_classification
from .classify_ms import classify_position_MS
from .regions import create_4_regions
from .AGB_and_BS_classification import AGB_BS_classification

from src.models import MainSequenceResults, Points


def classify_optical_data(df: pd.DataFrame, 
        CMD_config: int, 
        bins_division: int,
        main_sequence_results: MainSequenceResults,
        points: Points
        ) -> pd.DataFrame:
    
    """
    Classify optical sources into stellar populations based on their
    position in the Color-Magnitude Diagram (CMD).

    This function applies a multi-step classification pipeline:
    1) Classify sources relative to the Main Sequence (MS)
    2) Divide the CMD into four geometric regions
    3) Classify Horizontal Branch (HB) stars in region 4
    4) Classify Red Giant Branch (RGB), Red Stragglers (RS),
       and Red Supergiant Branch (SGB) in region 3
    5) Merge regions and classify AGB and Blue Straggler (BS) stars

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with photometric data.

    CMD_config : int
        Index defining the CMD configuration used.

    main_sequence_results : MainSequenceResults
        Results from main sequence detection, including:
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
        Reference points used in classification boundaries, including:
        (point_G : float, 
        point_H : float)
            

    Returns
    -------
    pd.DataFrame
        DataFrame with updated classification labels for all sources.
    """
    
    # Classify stars based on their position relative to the main sequence

    df = classify_position_MS(df, CMD_config, main_sequence_results)

    # Create regions based on CMD geometry

    region1, region2, region3, region4, points = create_4_regions(df, CMD_config, main_sequence_results, points)

    # Classify HB stars in region 4
    
    region4, points = HB_classification(region4, points, CMD_config)
    
    # Classify RGB, RS, and RSGB stars in region 3

    region3, points = RGB_classification(region3, CMD_config, bins_division, main_sequence_results.msto_y, points)

    # Merge all regions back together

    df_merged = pd.concat([region1, region2, region3, region4]).sort_index()

    # Classify AGB and BS stars in the whole CMD

    df_final = AGB_BS_classification(df_merged, CMD_config, points)

    return df_final