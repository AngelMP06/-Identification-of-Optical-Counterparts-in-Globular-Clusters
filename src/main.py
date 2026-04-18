from src.optical_pipeline.classify_optical import generate_optical_file
from src.xray_pipeline.classify_xray import creating_xray_csv

#generate_optical_file(distance_parsecs = 5400, 
#                      path_optical = 'C:/Users/HP/Desktop/Programación/Workspace/Python/Astronomy/NGC 6809/NGC6809.txt', 
#                      path_output = 'C:/Users/HP/Desktop/Programación/Workspace/Python/Astronomy/NGC 6809/NGC6809_optical.csv')

creating_xray_csv(
    path_xray="C:/Users/HP/Desktop/Programación/Workspace/Python/Astronomy/NGC 6809/sources.txt",
    path_output="C:/Users/HP/Desktop/Programación/Workspace/Python/Astronomy/NGC 6809/NGC6809_xray.csv",
    BS_RA = 295.0357771-295.0358042,
    BS_Decl = -30.9811814+30.9811333,
    distance_parsecs=5400
)