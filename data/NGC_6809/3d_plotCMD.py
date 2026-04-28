from matplotlib import colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import math
import matplotlib.cm as cm
from scipy.signal import savgol_filter
import json


path_optical = 'C:/Users/HP/Desktop/Curso de Python/Workspace/miweb/Astronomía/NGC 6809/NGC6809.txt'

distance_parsecs = 5400
Mag_min = -50  # 0
Mag_max = 50   # 35
RMS_max = 1     # 0.1 #No quiero que aparezcan varias estrellas dentro del circulo, pues si eso sucede de todas formas no será posible dar un valor correcto
Fit_max = 1.5   # También puede ser 0.6, pero en en los filtros 606 y 814 no se pueden usar valores tan estrictos, pues se eliminarían demasiados puntos
Sharp_min = -2  # -0.25
Sharp_max = 2   # 0.25
CM_min = 80

bins_division = 40
min_count = 2
min_step = 2

bluer_limit = 0.5
significantly_bluer_limit = 1.0
redder_limit = 0.5
significantly_redder_limit = 1.0


optical_data = pd.read_csv(
    path_optical,
    sep=r"\s+",      # split by any amount of whitespace
    header=None      # because there are no column names
)
optical_data.columns = [
        "X", "Y",
        "275_Mag", "275_RMS", "275_Fit", "275_Sharp", "275_exp_found", "275_exp_well",
        "336_Mag", "336_RMS", "336_Fit", "336_Sharp", "336_exp_found", "336_exp_well",
        "438_Mag", "438_RMS", "438_Fit", "438_Sharp", "438_exp_found", "438_exp_well",
        "606_Mag", "606_RMS", "606_Fit", "606_Sharp", "606_exp_found", "606_exp_well",
        "814_Mag", "814_RMS", "814_Fit", "814_Sharp", "814_exp_found", "814_exp_well",
        "Cluster_Membership",
        "RA", "Decl",
        "Id",
        "Iteration_found"
]


optical_data["Visible"] = (optical_data["438_Mag"]+optical_data["606_Mag"])/2

optical_data["Mv"] = optical_data["Visible"]-5*(math.log(distance_parsecs/10,10))

bands = ["275", "336", "438", "606", "814"]
mask = np.ones(len(optical_data), dtype=bool)
for b in bands:
    mask &= optical_data[f"{b}_Mag"].between(Mag_min, Mag_max)
    mask &= optical_data[f"{b}_RMS"] < RMS_max
    mask &= optical_data[f"{b}_Fit"] < Fit_max
    mask &= optical_data[f"{b}_Sharp"].between(Sharp_min, Sharp_max)
    mask &= optical_data["Cluster_Membership"] > CM_min


optical_data_filtered = optical_data[mask].copy()
optical_data_filtered.reset_index(inplace = True)
column_drop = ["X", "Y", "Iteration_found", "index"]
for i in bands:
    column_drop.append(i+"_RMS")
    column_drop.append(i+"_Fit")
    column_drop.append(i+"_Sharp")
    column_drop.append(i+"_exp_found")
    column_drop.append(i+"_exp_well")
optical_data_filtered.drop(columns = column_drop, inplace = True)

optical_data_filtered.sort_index(inplace = True)
optical_data_filtered['275 - 336'] = optical_data_filtered["275_Mag"] - optical_data_filtered["336_Mag"]
optical_data_filtered['438 - 606'] = optical_data_filtered["438_Mag"] - optical_data_filtered["606_Mag"]
optical_data_filtered['606 - 814'] = optical_data_filtered["606_Mag"] - optical_data_filtered["814_Mag"]


gráfico_comp = 1
#for gráfico_comp in range(3):
if gráfico_comp == 0: #F275W-F336W
    Color1 = "275"
    Color2 = "336"
elif gráfico_comp == 1: #F438W-F606W
    Color1 = "438"
    Color2 = "606"
elif gráfico_comp == 2: #F606W-F814W
    Color1 = "606"
    Color2 = "814"

#Def (optical_data_filtered, bins_division, S_dif, S_Color2)

S_Color1 = optical_data_filtered[Color1+"_Mag"]
S_Color2 = optical_data_filtered[Color2+"_Mag"]
S_dif = optical_data_filtered[f"{Color1} - {Color2}"]
n_Data = len(S_Color1)


#We define limits for the histogram, they have to be integer.
xmax = round(S_dif.max()+1)
xmin = round(S_dif.min()-1)
ymax = round(S_Color2.max()+1)
ymin = round(S_Color2.min()-1)

bins_xy = [bins_division*(xmax-xmin), bins_division*(ymax-ymin)]
range_xy = [[xmin, xmax], [ymin, ymax]]

H, x_edges, y_edges = np.histogram2d(S_dif, S_Color2, bins = bins_xy, range = range_xy)

for i in range(len(H)):
    for j in range(len(H[0])):
        if H[i][j] < min_count:
            H[i][j] = 0

cmap = plt.cm.viridis.copy()
cmap.set_under('white')  # values below vmin will be white

plt.pcolormesh(
    x_edges, y_edges, H.T,
    cmap=cmap,
    vmin=1  # everything <1 (i.e. 0) becomes white
)

x = []
min = []
max = []
for i in H.T:
    #First we go for every bin to find the position of the bins with stars, and we save the count of consecutive 
    # bins with stars (ou) and the count of consecutive bins without stars (n).
    l = []
    k = []
    ll = []
    kk = []
    ku = []
    kkuu = []
    ou = 0
    o = 0
    oo = 0
    n = 0
    c = 0
    for j,z in enumerate(i):
        #We consider that a bin has stars if its count is greater than 0, and we save the position and count of the 
        # bins with stars in k and kk, and we save the position and count of the last consecutive bins with stars in 
        # ku and kkuu.
        if z > 0:
            n = 0
            c = 0
            o += 1
            k.append(j)     #position
            kk.append(z)    #value
            ku = []
            kkuu = []
            ou = 0
        else:
            n += 1
            c += 2
            if n < 2:
                ku = k
                kkuu = kk
                ou = o
                o += 1
                k.append(j)     #position
                kk.append(z)    #value
            else:
                if ou > oo and ou >= min_step:
                    ll = kkuu
                    l = ku
                    oo = ou
                kk = []
                k = []
                o = 0
                ku = []
                kkuu = []
                ou = 0
    #We save the position of the left and right limits of the main sequence. If there are no stars in any bin, 
    # we save 0 as the limits.
    if len(l)>0:
        min.append(l[0])
        max.append(l[-1])
    else:
        min.append(0)
        max.append(0)
    total = 0
    #We also save the mean position of the bins with stars to plot the mean line of the main sequence.
    if len(l) > 0:
        for j in range(len(l)):
            total += l[j]*ll[j]
        x.append(total/sum(ll))
    else:
        x.append(0)


container = []
chain_width = 0
maximum = 0
for i,j in enumerate(x):
    if j == 0:
        container.append(i)
for i in range(len(container)-1):
    m = container[i+1]-container[i]-1
    if m > maximum:
        maximum = m
        chain_width = container[i]
main_sequence = range(chain_width+1,chain_width+maximum)
#Extract the positions (in cells) of the mean, max and min of the main sequence to plot them later. 
filtered_x = [x[i] for i in main_sequence]
filtered_max = [max[i] for i in main_sequence]
filtered_min = [min[i] for i in main_sequence]
#We obtain the position of the start of the main sequence turn off.
derivative_x = []
for i in range(len(x)-1):
    derivative_x.append(x[i]-x[i+1])
container_d = []
chain_width_d = 0
maximum_d = 0
for i,j in enumerate(derivative_x):
    if j <= 0:
        container_d.append(i)
for i in range(len(container_d)-1):
    m_d = container_d[i+1]-container_d[i]-1
    if m_d > maximum_d:
        maximum_d = m_d
        chain_width_d = container_d[i]
msto_start = chain_width_d+maximum_d
#Change the positions from cells to magnitude values of the CMD
normalized_min = (np.array(filtered_min)/bins_division)+xmin+1/(2*bins_division)
normalized_max = (np.array(filtered_max)/bins_division)+xmin-1/(2*bins_division)
#Smooth the lines of the minimum and maximum of the main sequence using the Savitzky-Golay
main_sequence_y_magnitude = np.array(main_sequence)/bins_division+ymin+1/(2*bins_division)
msto_start_y_magnitude = msto_start/bins_division+ymin+1/(2*bins_division)

window_size = 11
poly_order = 3
min_smooth = savgol_filter(normalized_min, window_size, poly_order)
max_smooth = savgol_filter(normalized_max, window_size, poly_order)
index_MSTO_y_start = np.where(main_sequence_y_magnitude == msto_start_y_magnitude)[0][0]
if index_MSTO_y_start < 5:
    index_SGB_y_start = 0
else:
    index_SGB_y_start = index_MSTO_y_start-5

plt.plot(min_smooth, main_sequence_y_magnitude, c = "red")
plt.plot(max_smooth, main_sequence_y_magnitude, c = "red")

plt.plot([S_dif.min(), S_dif.max()], [msto_start_y_magnitude, msto_start_y_magnitude], c = "red")
plt.plot([S_dif.min(), S_dif.max()], [main_sequence_y_magnitude[index_SGB_y_start], main_sequence_y_magnitude[index_SGB_y_start]], c = "green")

plt.gca().invert_yaxis()




plt.colorbar(label='Counts')
plt.show()


