import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NGC = "NGC6809"
gráfico_comp = 2
min_count = 2
min_step = 2

def create_3_graphs(NGC, sc):
    #Creación de los 3 gráficos
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (15,5))

    # Loading data
    path_optical = f'C:/Users/HP/Desktop/Curso de Python/Workspace/miweb/Astronomía/{NGC}_optical.csv'
    path_xray = f'C:/Users/HP/Desktop/Curso de Python/Workspace/miweb/Astronomía/{NGC}_xray.csv'


    # Data from all optical sources
    optical_data = pd.read_csv(path_optical)

    # Data from all x_rays
    xray_data = pd.read_csv(path_xray, index_col = "id")
    for gráfico_comp in range(3):
        if gráfico_comp == 0: #F275W-F336W
            Color1 = "275"
            Color2 = "336"
        elif gráfico_comp == 1: #F438W-F606W
            Color1 = "438"
            Color2 = "606"
        elif gráfico_comp == 2: #F606W-F814W
            Color1 = "606"
            Color2 = "814"
    
        #Data de los 2 colores que se van a comparar y de los datos generales
        df_Color1 = optical_data[[Color1+"_Mag"]]
        df_Color2 = optical_data[[Color2+"_Mag"]]
        df_Data = optical_data[["RA", "Decl", "Id"]]
    

        inner_circle_sources = []
    
        radio = "0.6"
    
        for i in range(len(df_Data)):
            #(RA-RA_centro)*cos(Decl_centro)**2 + (Decl-Decl_centro)**2 < r95**2
            if np.sqrt(((df_Data.iloc[i]["RA"]-xray_data.loc[sc]["RA"])*np.cos(df_Data.iloc[i]["Decl"]*np.pi/180))**2+(df_Data.iloc[i]["Decl"]-xray_data.loc[sc]["Decl"])**2) < 0.6/3600: #xray_data.loc[sc]["r95"]:
                inner_circle_sources.append(i)
        if len(inner_circle_sources) == 0:
            for i in range(len(df_Data)):
                if np.sqrt(((df_Data.iloc[i]["RA"]-xray_data.loc[sc]["RA"])*np.cos(df_Data.iloc[i]["Decl"]*np.pi/180))**2+(df_Data.iloc[i]["Decl"]-xray_data.loc[sc]["Decl"])**2) < 2/3600:
                    inner_circle_sources.append(i)
            radio = "2"
                
        dif=np.zeros(len(df_Data))
        Color1_array=np.zeros(len(df_Data))
        Color2_array=np.zeros(len(df_Data))
    
        for  i in range(len(df_Data)):
            Color1_array[i]=df_Color1.iloc[i][Color1+"_Mag"]
            Color2_array[i]=df_Color2.iloc[i][Color2+"_Mag"]
            dif[i]=Color1_array[i]-Color2_array[i]
        axes[gráfico_comp].scatter(dif, Color2_array, marker = ".", s=0.1, color='black')
        axes[gráfico_comp].invert_yaxis()
        axes[gráfico_comp].set_xlabel(Color1+"-"+Color2)
        axes[gráfico_comp].set_ylabel(Color2)
        axes[gráfico_comp].set_title(sc+" ("+Color1+"-"+Color2+") "+" radio: "+radio)
        
        dif2=np.zeros(len(inner_circle_sources))
        Color1_array2=np.zeros(len(inner_circle_sources))
        Color2_array2=np.zeros(len(inner_circle_sources))
    
        for  i,j in enumerate(inner_circle_sources):
            Color1_array2[i]=df_Color1.iloc[j][Color1+"_Mag"]
            Color2_array2[i]=df_Color2.iloc[j][Color2+"_Mag"]
            dif2[i]=Color1_array2[i]-Color2_array2[i]
    
        for n in range(len(inner_circle_sources)):
            axes[gráfico_comp].scatter(dif2[n],Color1_array2[n],marker="o",s=30+9*n,label=df_Data.iloc[inner_circle_sources[n]]["Id"])
        axes[gráfico_comp].legend()
    plt.savefig('C:/Users/HP/Desktop/Curso de Python/Workspace/miweb/Astronomía/Imágenes/'+sc+'.png', dpi=400) # You can adjust the dpi for resolution
    plt.show()

def graph_xray_class(NGC):

    path_xray = f'C:/Users/HP/Desktop/Curso de Python/Workspace/miweb/Astronomía/{NGC}_xray.csv'
    df = pd.read_csv(path_xray, index_col = "id")

    x = np.linspace(-1, 2, 2)
    a = [-0.1, 0.5]
    b = [34, 31.6020599913]
    plt.scatter(df["log(Lx_0.5-2.5 / Lx_2.5-6.0)"], df["log(Lx_0.5-6.0)"])
    for i in range(len(df["log(Lx_0.5-6.0)"])):
        plt.annotate(df ["Hardness_classification"].iloc[i]+"("+df.index[i]+")", (df["log(Lx_0.5-2.5 / Lx_2.5-6.0)"].iloc[i], df["log(Lx_0.5-6.0)"].iloc[i]))
    plt.plot(x, [30.6020599913, 30.6020599913])
    plt.plot(x, [31.6020599913, 31.6020599913])
    plt.plot(a,b)
    plt.show()


create_3_graphs("NGC6809", "CX8")

