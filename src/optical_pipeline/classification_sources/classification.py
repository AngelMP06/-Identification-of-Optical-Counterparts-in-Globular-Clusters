import pandas as pd

from .RGB_classificacion import RGB_classification
from .HB_classification import HB_classification
from .classify_ms import classify_position_MS
from .regions import create_4_regions
from .AGB_and_BS_classification import AGB_BS_classification

def classify_optical_data(df, 
        graphic_comp, 
        main_sequence_y, 
        min_smooth, 
        max_smooth, 
        index_SGB, 
        index_MSTO, 
        significantly_bluer_limit, 
        bluer_limit, 
        redder_limit, 
        significantly_redder_limit, 
        bins_division, 
        msto_y,
        point_G,
        point_H
        ):
    
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
    graphic_comp : int
        Index defining the CMD configuration used.
    main_sequence_y, min_smooth, max_smooth : array-like
        Main Sequence ridge and boundaries.
    index_SGB, index_MSTO : int
        Indices marking evolutionary transitions.
    significantly_bluer_limit, bluer_limit, redder_limit, significantly_redder_limit : float
        Thresholds for MS-based classification.
    bins_division : int
        Resolution parameter for histogram-based methods.
    msto_y : float
        Magnitude of the Main Sequence Turn-Off.
    point_G, point_H : float
        Reference points used in classification boundaries.

    Returns
    -------
    pd.DataFrame
        DataFrame with updated classification labels for all sources.
    """
    
    # Classify stars based on their position relative to the main sequence

    df = classify_position_MS(
            df,
            graphic_comp,
            main_sequence_y,
            min_smooth,
            max_smooth,
            index_SGB,
            index_MSTO,
            significantly_bluer_limit,
            bluer_limit,
            redder_limit,
            significantly_redder_limit
            )

    # Create regions based on CMD geometry

    region1, region2, region3, region4 = create_4_regions(
                                            df,
                                            graphic_comp,
                                            main_sequence_y,
                                            msto_y,
                                            min_smooth
                                            )

    # Classify HB stars in region 4

    region4, point_C, point_D, point_E = HB_classification(
                                            region4,
                                            graphic_comp,
                                            eps=0.4,
                                            min_samples=15
                                            )
    # Classify RGB, RS, and RSGB stars in region 3

    region3, point_F = RGB_classification(
                        region3,
                        graphic_comp,
                        bins_division,
                        msto_y,
                        eps=0.4,
                        min_samples=15
                        )
    
    # Merge all regions back together

    df_merged = pd.concat([region1, region2, region3, region4]).sort_index()

    # Classify AGB and BS stars in the whole CMD

    df_final = AGB_BS_classification(
                df_merged,
                graphic_comp,
                point_C, point_D, point_E, point_F, point_G, point_H  
                )

    return df_final