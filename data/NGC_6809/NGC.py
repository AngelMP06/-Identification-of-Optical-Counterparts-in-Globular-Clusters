import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
NGC = "NGC6809"
cx_list = [f"CX{i}" for i in range(1, 32)]

path_optical = f'C:/Users/HP/Desktop/Curso de Python/Workspace/miweb/Astronomía/{NGC}_optical.csv'
path_xray = f'C:/Users/HP/Desktop/Curso de Python/Workspace/miweb/Astronomía/{NGC}_xray.csv'


    # Data from all optical sources
optical_data = pd.read_csv(path_optical)

    # Data from all x_rays
xray_data = pd.read_csv(path_xray, index_col = "id")

data_list = []

for CX in cx_list:
    df_Data = optical_data[["RA", "Decl", "Id"]]
    ra = df_Data["RA"].values
    dec = df_Data["Decl"].values    
    ra0 = xray_data.loc[CX, "RA"]
    dec0 = xray_data.loc[CX, "Decl"]
    
    radius = "0.6"        
    dist = np.sqrt(
        ((ra - ra0) * np.cos(np.deg2rad(dec)))**2 +
        (dec - dec0)**2
    )
    inner_circle_sources = np.where(dist < 0.6/3600)[0].tolist()
            
    if len(inner_circle_sources) == 0:
        inner_circle_sources = np.where(dist < 2/3600)[0].tolist()
        radius = "2"
    n = optical_data.loc[inner_circle_sources]
    
    for num, sources in enumerate(inner_circle_sources):
        Mv = optical_data.loc[sources]["Mv"]
        xray_L = xray_data.loc[CX]["log(Lx_0.5-6.0)"]
        zzz = 0.4 * Mv + xray_L
        if zzz < 34:
            optical_xray_classification = "AB"
        elif zzz >= 34 and zzz < 36.2:
            optical_xray_classification = "CV"
        else:
            optical_xray_classification = "LMXRB"
        new_row_data = {'id': f"{CX}_{num+1}",
                        'Hardness_classification': xray_data.loc[CX]["Hardness_classification"],
                        "Xray source" : CX,
                        'Mv vs xray': optical_xray_classification,
                        'pos_0' : optical_data.loc[sources]["position_0"],
                        '275 - 336' : optical_data.loc[sources]["275 - 336"],
                        '336_Mag' : optical_data.loc[sources]["336_Mag"],
                        'pos_1' : optical_data.loc[sources]["position_1"],
                        '438 - 606' : optical_data.loc[sources]["438 - 606"],
                        '606_Mag' : optical_data.loc[sources]["606_Mag"],
                        'pos_2' : optical_data.loc[sources]["position_2"],
                        '606 - 814' : optical_data.loc[sources]["606 - 814"],
                        '814_Mag' : optical_data.loc[sources]["814_Mag"],
                        'radius' : radius,
                        '#sources' : len(inner_circle_sources)}
        data_list.append(new_row_data)

with open("C:/Users/HP/Desktop/Curso de Python/Workspace/miweb/Astronomía/data_obtained.txt", "w") as f:
        json.dump(data_list, f, indent=4)

df = pd.DataFrame(data_list)

fig, axes = plt.subplots(ncols = 3, figsize=(20, 10))

pairs = [("275", "336"), ("438", "606"), ("606", "814")]

for gráfico_comp in range(3):

    Color1, Color2 = pairs[gráfico_comp]

    sns.scatterplot(
        data=optical_data,
        x=f"{Color1} - {Color2}",
        y=f"{Color2}_Mag",
#        hue="position_0",
        color = "black",
        s=1,
        legend=False,
        ax=axes[gráfico_comp]
    )

    for i in range(12):
        axes[gráfico_comp].scatter(
            df.loc[i][f"{Color1} - {Color2}"],
            df.loc[i][f"{Color2}_Mag"],
            s=20+i*5,
            label=df.loc[i]["id"]
        )

    axes[gráfico_comp].invert_yaxis()
    axes[gráfico_comp].legend()

plt.show()