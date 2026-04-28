from src.optical_pipeline.classification_sources.classification import classify_optical_data
from src.optical_pipeline.preprocessing.load_data import load_optical_data
from src.optical_pipeline.preprocessing.filter_data import filter_data
from src.optical_pipeline.main_sequence_detection.find_main_sequence import find_main_sequence

def optical_classification(path_optical, distance_parsecs, 
                          Mag_min = -50, Mag_max = 50, RMS_max = 1, Fit_max = 1.5, 
                          Sharp_min = -2, Sharp_max = 2, CM_min = 80, bins_division = 20, 
                          min_count = 2, min_step = 2, maximum_lenght_MSTO = 0.70,
                          bluer_limit = 0.5, significantly_bluer_limit = 1.0, redder_limit = 0.5, significantly_redder_limit = 1.0):
    """
    Classify the sources according to their position in the CMD.

    Returns:
        source_data: DataFrame
    """

    # Step 1: We load all the optical data we need and assign the column names.

    optical_data = load_optical_data(path_optical, distance_parsecs)

    # Step 2: Filter data based on criteria

    source_data = filter_data(optical_data, Mag_min, Mag_max, RMS_max, Fit_max, Sharp_min, Sharp_max, CM_min)

    for graphic_comp in range(3):
        source_data[f"position_{graphic_comp}"] = "Unknown"

        # Step 3: We now are going to find the left and right edge of the main sequence and main sequence turn off (MSTO).

        main_sequence_y, min_smooth, max_smooth, index_SGB, index_MSTO, point_G, point_I, msto_y = find_main_sequence(source_data, 
                                                                                            bins_division, 
                                                                                            graphic_comp, 
                                                                                            min_count,     
                                                                                            min_step,
                                                                                            maximum_lenght_MSTO)
        # Step 4: Classify stars based on their position relative to the main sequence

        df_classified = classify_optical_data(source_data,
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
                                point_I
                                )

        column = f"position_{graphic_comp}"
        source_data[column] = df_classified[column]

    print("Optical classification completed")

    return source_data


