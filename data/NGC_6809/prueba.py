import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

ruta = 'C:/Users/HP/Desktop/Curso de Python/Recursos/Día 11/Cuadernos para Prácticas/Ventas.csv'
df = pd.read_csv(ruta)

fechas = pd.to_datetime(df["Fecha"])
plt.plot(fechas, df["Ventas"])
plt.show()