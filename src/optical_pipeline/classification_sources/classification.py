import pandas as pd

from .GB_classificacion import GB_classification
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

    # Classify HB stars in region 4 (which contains the HB)

    region4, point_C, point_D, point_E = HB_classification(
                                            region4,
                                            graphic_comp,
                                            eps=0.4,
                                            min_samples=15
                                            )


    region3, point_F = GB_classification(
                        region3,
                        graphic_comp,
                        bins_division,
                        msto_y,
                        eps=0.4,
                        min_samples=15
                        )
    
    # Merge all regions back together

    df_merged = pd.concat([region1, region2, region3, region4]).sort_index()
    
    col = f"position_{graphic_comp}"

    mask = df_merged[col].isin([
        "redder than SGB",
        "redder than MSTO"
    ])

    df_merged.loc[mask, col] = "Sub-Sub-Giant Branch"

    df_final = AGB_BS_classification(
    df_merged,
    graphic_comp,
    point_C, point_D, point_E, point_F, point_G, point_H
    )

    return df_final